# ==========================================
# [Final v35.0] 태풍 분석 통합 시스템 (Hybrid Offline Edition)
# ==========================================
# 특징: 고정DB 자동 로드 + 변동 파일 2개 업로드 + 보안 완벽 + 결과 엑셀 다운로드
import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import folium
from folium.features import DivIcon
from datetime import datetime, timedelta, time
import airportsdata
import io
import os

# ---------------------------------------------------------
# [STREAMLIT CONFIG]
# ---------------------------------------------------------
st.set_page_config(page_title="Typhoon Flight Analyzer (Hybrid)", layout="wide", page_icon="✈️")

# 사용자 설정
with st.sidebar:
    st.header("⚙️ 엔진 설정")
    USE_INTERPOLATION = st.checkbox("내삽(Interpolation) 사용", value=True, help="체크 시 정밀도 상승, 해제 시 속도 상승")
    MAX_VALID_SEGMENT_NM = st.number_input("점프 방지 거리(nm)", value=600, help="이 거리 이상 벌어지면 데이터 오류로 간주 (LJG 등 내륙 점프 방지)")
    st.markdown("---")
    st.info("💡 **고정 DB (Waypoint, Airway, DB_ROUTE)**는 시스템 폴더에서 자동으로 불러옵니다.")

# ---------------------------------------------------------
# 1. 고정 데이터 & 유틸리티 캐싱 (속도 최적화)
# ---------------------------------------------------------
@st.cache_resource
def load_airports():
    return airportsdata.load('iata'), airportsdata.load('icao')

airports_iata, airports_icao = load_airports()

# 🗄️ [핵심] 고정 DB 3개 파일 읽어오기 (앱 실행 시 최초 1회만 메모리에 적재)
@st.cache_data
def load_static_db():
    try:
        # 현재 폴더에 있는 엑셀 파일들을 읽어옵니다. (파일명이 정확히 일치해야 함)
        wp_raw = pd.read_excel("Waypoint.xlsx")
        aw_raw = pd.read_excel("airway.xlsx")
        rte_raw = pd.read_excel("DB_ROUTE.xlsx")
        
        wp_df = wp_raw.dropna(subset=[wp_raw.columns[0]])
        aw_df = aw_raw.dropna(subset=[aw_raw.columns[0]])
        route_df = rte_raw.dropna(how='all')
        
        return wp_df, aw_df, route_df
    except FileNotFoundError as e:
        return None, None, None

def get_codes(code):
    c = str(code).strip().upper()
    res = {c}
    if len(c)==3 and c in airports_iata: res.add(airports_iata[c]['icao'])
    elif len(c)==4 and c in airports_icao: res.add(airports_icao[c]['iata'])
    return list(res)

def get_airport_coords(code):
    c = str(code).strip().upper()
    if len(c)==4 and c in airports_icao: return (airports_icao[c]['lat'], airports_icao[c]['lon'])
    if len(c)==3 and c in airports_iata: return (airports_iata[c]['lat'], airports_iata[c]['lon'])
    return None

def dms_to_decimal(dms_str):
    if isinstance(dms_str, (int, float)): return float(dms_str)
    s = str(dms_str).strip().upper()
    try:
        sign = -1 if 'S' in s or 'W' in s else 1
        clean_s = re.sub(r'[NSEW]', '', s)
        parts = re.split(r'[-:\s]+', clean_s)
        d = float(parts[0]) if len(parts)>0 else 0
        m = float(parts[1]) if len(parts)>1 else 0
        sec = float(parts[2]) if len(parts)>2 else 0
        return sign * (d + m/60 + sec/3600)
    except: return None

def is_valid_coord(p):
    if not p: return False
    try:
        if pd.isna(p[0]) or pd.isna(p[1]): return False
        if p[0] == 0 and p[1] == 0: return False
        if not (-90 <= p[0] <= 90) or not (-180 <= p[1] <= 180): return False
        return True
    except: return False

def fast_dist_nm(p1, p2):
    if not is_valid_coord(p1) or not is_valid_coord(p2): return 99999.0
    try:
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        dlat = lat2 - lat1; dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return 3440.065 * c
    except: return 99999.0

def interpolate_segment(p1, p2, interval_nm=50):
    if not USE_INTERPOLATION: return []
    dist = fast_dist_nm(p1, p2)
    if dist >= 2000 or dist <= interval_nm: return []
    try:
        num_points = int(dist // interval_nm)
        points = []
        for i in range(1, num_points + 1):
            f = i / (num_points + 1)
            lat = p1[0] + (p2[0] - p1[0]) * f
            lon = p1[1] + (p2[1] - p1[1]) * f
            points.append((lat, lon))
        return points
    except: return []

# ---------------------------------------------------------
# 2. 엔진 로직 (캐시된 데이터프레임 사용)
# ---------------------------------------------------------
class HybridEngine:
    def __init__(self, wp_df, aw_df, route_df):
        self.wp_df = wp_df
        self.airway_df = aw_df
        self.db_route_df = route_df
        self.global_db = {}
        self.airway_dict = {}
        self.route_cache = {}

    def build_db(self):
        # Waypoints
        names = self.wp_df.iloc[:, 0].astype(str).str.strip().str.upper().values
        lats = [dms_to_decimal(x) for x in self.wp_df.iloc[:, 3].values]
        lons = [dms_to_decimal(x) for x in self.wp_df.iloc[:, 4].values]
        for n, lat, lon in zip(names, lats, lons):
            if is_valid_coord((lat, lon)): 
                self.global_db.setdefault(n, []).append((lat, lon))
        
        # Airway Points
        ids = self.airway_df.iloc[:, 0].fillna("").astype(str).str.strip().str.upper().values
        names_aw = self.airway_df.iloc[:, 2].astype(str).str.strip().str.upper().values
        lats_aw = [dms_to_decimal(x) for x in self.airway_df.iloc[:, 4].values]
        lons_aw = [dms_to_decimal(x) for x in self.airway_df.iloc[:, 5].values]
        for aid, name, lat, lon in zip(ids, names_aw, lats_aw, lons_aw):
            if not aid or not is_valid_coord((lat, lon)): continue
            self.global_db.setdefault(name, []).append((lat, lon))
            if aid not in self.airway_dict: self.airway_dict[aid] = []
            self.airway_dict[aid].append({'name': name, 'coord': (lat, lon)})

    def get_route_data(self, route_name, strip, dep, arr):
        cache_key = f"{route_name}_{dep}_{arr}"
        if cache_key in self.route_cache: return self.route_cache[cache_key]
        coords = self._build_route_raw(strip, dep, arr)
        if not coords: return None
        total_dist = 0; seg_dists = [0]
        for i in range(len(coords)-1):
            d = fast_dist_nm(coords[i], coords[i+1])
            total_dist += d; seg_dists.append(seg_dists[-1] + d)
        data = {'coords': coords, 'total_dist': total_dist, 'seg_dists': seg_dists}
        self.route_cache[cache_key] = data
        return data

    def _build_route_raw(self, strip, dep, arr):
        tokens = re.split(r'[\s\.,]+', str(strip))
        tokens = [t.strip().upper() for t in tokens if t.strip()]
        coords = []; dep_c = get_airport_coords(dep); arr_c = get_airport_coords(arr)
        if dep_c: coords.append(dep_c)
        prev = coords[-1] if coords else None
        
        for i, t in enumerate(tokens):
            if t in self.airway_dict:
                aw = self.airway_dict[t]
                if not prev: continue
                best_s = min(range(len(aw)), key=lambda k: fast_dist_nm(prev, aw[k]['coord']))
                dest = arr_c
                if i+1 < len(tokens):
                    nxt = tokens[i+1]
                    matches = [k for k, p in enumerate(aw) if p['name'] == nxt]
                    if matches: dest = aw[matches[0]]['coord']
                if not dest: continue
                best_e = min(range(len(aw)), key=lambda k: fast_dist_nm(dest, aw[k]['coord']))
                
                if best_s <= best_e: raw = aw[best_s:best_e+1]
                else: raw = aw[best_e:best_s+1][::-1]
                
                for pt in raw:
                    curr = pt['coord']
                    if fast_dist_nm(prev, curr) > MAX_VALID_SEGMENT_NM: continue
                    coords.extend(interpolate_segment(prev, curr, 50)); coords.append(curr); prev = curr
            else:
                cand = self.global_db.get(t)
                if cand:
                    sel = min(cand, key=lambda p: fast_dist_nm(p, arr_c) if arr_c else 0)
                    if prev:
                        if fast_dist_nm(prev, sel) > MAX_VALID_SEGMENT_NM: continue
                        coords.extend(interpolate_segment(prev, sel, 50))
                    coords.append(sel); prev = sel
        return coords

# ---------------------------------------------------------
# UI 화면 구성
# ---------------------------------------------------------
st.title("🌪️ Typhoon Flight Analyzer (Hybrid Edition)")

# 1. 고정 DB 로드 상태 확인
wp_df, aw_df, route_df = load_static_db()

if wp_df is None or aw_df is None or route_df is None:
    st.error("🚨 폴더에 고정 데이터베이스 파일 3개가 없습니다. (`Waypoint.xlsx`, `airway.xlsx`, `DB_ROUTE.xlsx`)")
    st.info("이 앱을 구동하는 폴더(또는 GitHub)에 위 3개의 엑셀 파일을 정확한 이름으로 업로드해주세요.")
    st.stop()
else:
    if 'engine' not in st.session_state:
        with st.spinner("📦 고정 데이터베이스 초기화 중... (최초 1회)"):
            st.session_state.engine = HybridEngine(wp_df, aw_df, route_df)
            st.session_state.engine.build_db()
        st.success("✅ 고정 데이터베이스 로드 완료! (가장 무거운 작업이 끝났습니다)")

st.markdown("### 🔄 오늘의 분석 파일 업로드")
st.markdown("매일 바뀌는 스케줄과 태풍 정보 엑셀 파일만 업로드해주세요.")

col1, col2 = st.columns(2)
with col1:
    f_skd = st.file_uploader("✈️ SKD_BASE (오늘의 스케줄)", type=['xlsx'])
with col2:
    f_rest = st.file_uploader("🌪️ Restrictions (오늘의 태풍 정보)", type=['xlsx'])

if f_skd and f_rest:
    if st.button("🚀 비행편 분석 시작", type="primary", use_container_width=True):
        with st.spinner("운항편 정밀 분석 중..."):
            eng = st.session_state.engine
            skd_df = pd.read_excel(f_skd)
            rest_df = pd.read_excel(f_rest)
            
            # 태풍 파싱
            typhoons = []
            for _, r in rest_df.iterrows():
                try:
                    parts = re.split(r'[,\s]+', str(r.iloc[1]).strip())
                    typhoons.append({
                        'n': r.iloc[0], 'c': (float(parts[0]), float(parts[1])),
                        's': pd.to_datetime(r.iloc[2]), 'e': pd.to_datetime(r.iloc[3]), 'r': float(r.iloc[4])
                    })
                except: continue
            
            res_list = []; progress_bar = st.progress(0); status_text = st.empty()
            
            for idx, row in skd_df.iterrows():
                progress_bar.progress((idx + 1) / len(skd_df))
                try:
                    # 스케줄 엑셀 열 인덱스 가정 (1=FLT, 2=DATE, 8=DEP, 9=ARR, 10=STD, 11=STA)
                    f_no = str(row.iloc[1]); dep = str(row.iloc[8]).strip(); arr = str(row.iloc[9]).strip()
                    status_text.text(f"분석 중: {f_no} ({dep}->{arr})")
                    
                    matched_routes = eng.db_route_df[
                        (eng.db_route_df.iloc[:,0].astype(str).str.upper().str.contains(dep)) & 
                        (eng.db_route_df.iloc[:,1].astype(str).str.upper().str.contains(arr))
                    ]
                    
                    if matched_routes.empty: continue
                    
                    d_raw = pd.to_datetime(row.iloc[2], errors='coerce')
                    if pd.isna(d_raw): continue
                    
                    try:
                        t_std = pd.to_datetime(str(row.iloc[10])).time()
                        t_sta = pd.to_datetime(str(row.iloc[11])).time()
                    except: continue

                    dt_std = datetime.combine(d_raw.date(), t_std)
                    dt_sta = datetime.combine(d_raw.date(), t_sta)
                    if dt_sta < dt_std: dt_sta += timedelta(days=1)
                    fly_hours = (dt_sta - dt_std).total_seconds() / 3600 - 0.5
                    if fly_hours < 0.5: fly_hours = 0.5

                    # [P/W 필터]
                    route_objs = []
                    for _, r in matched_routes.iterrows():
                        r_name = str(r.iloc[2]).strip().upper()
                        if not (r_name.startswith('P') or r_name.startswith('W')): continue
                        r_data = eng.get_route_data(r_name, str(r.iloc[4]), dep, arr)
                        if r_data: route_objs.append({'name': r_name, 'data': r_data})
                        
                    if not route_objs: continue
                    
                    # [P01 속도 기준 로직]
                    ref_route = next((r for r in route_objs if 'P01' in r['name'].upper()), None)
                    if not ref_route: ref_route = min(route_objs, key=lambda x: x['data']['total_dist'])
                    avg_speed = ref_route['data']['total_dist'] / fly_hours
                    if avg_speed < 100: avg_speed = 450.0 
                    
                    risk_routes = []
                    safe_list = []
                    
                    for r in route_objs:
                        r_name = r['name']; r_data = r['data']
                        est_hours = r_data['total_dist'] / avg_speed
                        est_fly_time = timedelta(hours=est_hours)
                        
                        hit = False; hit_msg = ""
                        for ty in typhoons:
                            for k, pt in enumerate(r_data['coords']):
                                if fast_dist_nm(pt, ty['c']) <= ty['r']:
                                    progress = r_data['seg_dists'][k] / r_data['total_dist']
                                    p_time = dt_std + timedelta(minutes=15) + (est_fly_time * progress)
                                    if ty['s'] <= p_time <= ty['e']:
                                        hit = True; hit_msg = f"{r_name}({ty['n']})"
                                        break
                            if hit: break
                        
                        if hit: risk_routes.append(hit_msg)
                        else: safe_list.append({'name': r_name, 'dist': r_data['total_dist']})
                    
                    if risk_routes:
                        safe_list.sort(key=lambda x: x['dist'])
                        res_list.append({
                            'FLT': f_no, 'DATE': str(d_raw.date()), 'DEP': dep, 'ARR': arr,
                            'STD': t_std.strftime("%H:%M"), 'STA': t_sta.strftime("%H:%M"),
                            'RESTRICTED_ROUTES': ", ".join(risk_routes),
                            'REC_ROUTE_1': safe_list[0]['name'] if len(safe_list)>0 else "N/A",
                            'DIST_1': f"{safe_list[0]['dist']:.0f}" if len(safe_list)>0 else "",
                            'REC_ROUTE_2': safe_list[1]['name'] if len(safe_list)>1 else "",
                            'DIST_2': f"{safe_list[1]['dist']:.0f}" if len(safe_list)>1 else "",
                            'REC_ROUTE_3': safe_list[2]['name'] if len(safe_list)>2 else "",
                            'DIST_3': f"{safe_list[2]['dist']:.0f}" if len(safe_list)>2 else "",
                            'REC_ROUTE_4': safe_list[3]['name'] if len(safe_list)>3 else "",
                            'DIST_4': f"{safe_list[3]['dist']:.0f}" if len(safe_list)>3 else ""
                        })
                except Exception as ex: 
                    continue

            status_text.empty()
            if res_list:
                df_res = pd.DataFrame(res_list)
                st.success(f"🔥 총 {len(df_res)}건의 제한 운항편이 발견되었습니다!")
                st.dataframe(df_res)
                
                # 엑셀 다운로드 포맷 생성 (Summary + 날짜별 시트 분리)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_res.to_excel(writer, index=False, sheet_name='Summary')
                    for d in df_res['DATE'].unique():
                        sub_df = df_res[df_res['DATE'] == d]
                        sub_df.to_excel(writer, index=False, sheet_name=f"RES_{d}")
                
                st.download_button(label="💾 최종 분석 결과 엑셀 다운로드", data=output.getvalue(), file_name="Typhoon_Analysis_Result.xlsx", mime="application/vnd.ms-excel")
            else:
                st.success("✅ 태풍의 영향을 받는 제한 운항편이 없습니다.")