# ==========================================
# [Final v35.4] 태풍 분석 통합 시스템 (Dual Core + 정밀 에어웨이 추적 부활)
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from datetime import datetime, timedelta, time
import airportsdata
import io

# ---------------------------------------------------------
# [STREAMLIT CONFIG]
# ---------------------------------------------------------
st.set_page_config(page_title="Typhoon Flight Analyzer", layout="wide", page_icon="✈️")

with st.sidebar:
    st.header("⚙️ 엔진 설정")
    USE_INTERPOLATION = st.checkbox("내삽(Interpolation) 사용", value=True)
    MAX_VALID_SEGMENT_NM = st.number_input("점프 방지 거리(nm)", value=600)
    st.markdown("---")
    st.info("💡 **정밀 에어웨이 알고리즘(v31.0)** 및 **중국 영공 통과 추적기**가 활성화되었습니다.")

# ---------------------------------------------------------
# 1. 고정 데이터 & 유틸리티
# ---------------------------------------------------------
@st.cache_resource
def load_airports():
    return airportsdata.load('iata'), airportsdata.load('icao')

airports_iata, airports_icao = load_airports()

@st.cache_data
def load_static_db():
    try:
        wp_raw = pd.read_excel("Waypoint.xlsx")
        aw_raw = pd.read_excel("airway.xlsx")
        rte_raw = pd.read_excel("DB_ROUTE.xlsx")
        
        try:
            fix_raw = pd.read_csv("FIX_result.csv")
        except:
            fix_raw = None
        
        wp_df = wp_raw.dropna(subset=[wp_raw.columns[0]])
        aw_df = aw_raw.dropna(subset=[aw_raw.columns[0]])
        route_df = rte_raw.dropna(how='all')
        
        return wp_df, aw_df, route_df, fix_raw
    except FileNotFoundError:
        return None, None, None, None

def parse_wkt_point(wkt_str):
    try:
        match = re.match(r'POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)', str(wkt_str).upper())
        if match:
            lon, lat = float(match.group(1)), float(match.group(2))
            return (lat, lon)
    except: pass
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

# ---------------------------------------------------------
# 2. 듀얼 코어 & 중국 FIR 판별 엔진
# ---------------------------------------------------------
class DualCoreEngine:
    def __init__(self, wp_df, aw_df, route_df, fix_df):
        self.wp_df = wp_df; self.airway_df = aw_df; self.db_route_df = route_df; self.fix_df = fix_df
        self.global_db = {}; self.airway_dict = {}; self.route_cache = {}
        self.china_nodes = set()

    def build_db(self):
        seen_coords = set()
        
        # [코어 1] GIS 데이터
        if self.fix_df is not None:
            names_fix = self.fix_df.iloc[:, 0].astype(str).str.strip().str.upper().values
            fir_fix = self.fix_df.iloc[:, 4].astype(str).str.strip().str.upper().values
            points_fix = self.fix_df.iloc[:, 5].astype(str).values
            
            for n, fir, pt_str in zip(names_fix, fir_fix, points_fix):
                coord = parse_wkt_point(pt_str)
                if coord and is_valid_coord(coord):
                    approx = (round(coord[0], 2), round(coord[1], 2))
                    if (n, approx) not in seen_coords:
                        self.global_db.setdefault(n, []).append(coord)
                        seen_coords.add((n, approx))
                    if fir.startswith('Z') and not fir.startswith('ZK') and not fir.startswith('ZM'):
                        self.china_nodes.add((n, approx))

        # [코어 2] Waypoint.xlsx 데이터
        names_wp = self.wp_df.iloc[:, 0].astype(str).str.strip().str.upper().values
        lats_wp = [dms_to_decimal(x) for x in self.wp_df.iloc[:, 3].values]
        lons_wp = [dms_to_decimal(x) for x in self.wp_df.iloc[:, 4].values]
        ccs_wp = self.wp_df.iloc[:, 6].astype(str).str.strip().str.upper().values
        
        for n, lat, lon, cc in zip(names_wp, lats_wp, lons_wp, ccs_wp):
            if is_valid_coord((lat, lon)):
                approx = (round(lat, 2), round(lon, 2))
                if (n, approx) not in seen_coords:
                    self.global_db.setdefault(n, []).append((lat, lon))
                    seen_coords.add((n, approx))
                if cc.startswith('Z') and not cc.startswith('ZK') and not cc.startswith('ZM'):
                    self.china_nodes.add((n, approx))
        
        # [Airway 로드]
        ids = self.airway_df.iloc[:, 0].fillna("").astype(str).str.strip().str.upper().values
        names_aw = self.airway_df.iloc[:, 2].astype(str).str.strip().str.upper().values
        lats_aw = [dms_to_decimal(x) for x in self.airway_df.iloc[:, 4].values]
        lons_aw = [dms_to_decimal(x) for x in self.airway_df.iloc[:, 5].values]
        
        for aid, name, lat, lon in zip(ids, names_aw, lats_aw, lons_aw):
            if not aid or not is_valid_coord((lat, lon)): continue
            approx = (round(lat, 2), round(lon, 2))
            if (name, approx) not in seen_coords:
                self.global_db.setdefault(name, []).append((lat, lon))
                seen_coords.add((name, approx))
            if aid not in self.airway_dict: self.airway_dict[aid] = []
            self.airway_dict[aid].append({'name': name, 'coord': (lat, lon)})

    def get_route_data(self, route_name, strip, dep, arr):
        cache_key = f"{route_name}_{dep}_{arr}"
        if cache_key in self.route_cache: return self.route_cache[cache_key]
        
        coords_info = self._build_route_raw(strip, dep, arr)
        if not coords_info: return None
        
        coords = [pt['coord'] for pt in coords_info]
        total_dist = 0; seg_dists = [0]
        for i in range(len(coords)-1):
            d = fast_dist_nm(coords[i], coords[i+1])
            total_dist += d; seg_dists.append(seg_dists[-1] + d)
            
        data = {'coords': coords, 'info': coords_info, 'total_dist': total_dist, 'seg_dists': seg_dists}
        self.route_cache[cache_key] = data
        return data

    # 🚨 v31.0의 가장 강력한 '정밀 에어웨이 알고리즘' 완벽 부활!
    def _build_route_raw(self, strip, dep, arr):
        tokens = re.split(r'[\s\.,]+', str(strip))
        tokens = [t.strip().upper() for t in tokens if t.strip()]
        coords_info = []
        dep_keys, arr_keys = get_codes(dep), get_codes(arr)
        
        final_dest = get_airport_coords(arr_keys[0]) if arr_keys else None
        if not final_dest and len(arr_keys)>1: final_dest = get_airport_coords(arr_keys[-1])
        start_c = get_airport_coords(dep_keys[0]) if dep_keys else None
        if not start_c and len(dep_keys)>1: start_c = get_airport_coords(dep_keys[-1])
        
        if start_c: 
            is_cn = any(len(k)==4 and k.startswith('Z') and not k.startswith('ZK') and not k.startswith('ZM') for k in dep_keys)
            coords_info.append({'coord': start_c, 'name': dep, 'is_china': is_cn})
        prev_coord = start_c if start_c else None
        
        for i, t in enumerate(tokens):
            if t in self.airway_dict:
                if not prev_coord: continue
                aw = self.airway_dict[t]
                s_indices = []
                for idx, pt in enumerate(aw):
                    if fast_dist_nm(prev_coord, pt['coord']) < 5.0: s_indices.append(idx)
                if not s_indices:
                    min_d = 999999; best_s = 0
                    for idx, pt in enumerate(aw):
                        d = fast_dist_nm(prev_coord, pt['coord'])
                        if d < min_d: min_d = d; best_s = idx
                    s_indices.append(best_s)
                
                e_indices = []
                if i+1 < len(tokens):
                    nxt = tokens[i+1]
                    name_matches = [idx for idx, pt in enumerate(aw) if pt['name'] == nxt]
                    if not name_matches:
                        target_coord = None
                        if get_airport_coords(nxt): target_coord = get_airport_coords(nxt)
                        elif nxt in self.global_db:
                            target_coord = min(self.global_db[nxt], key=lambda p: fast_dist_nm(p, final_dest) if final_dest else 0)
                        if target_coord:
                            min_err = 5.0
                            for idx, pt in enumerate(aw):
                                d = fast_dist_nm(target_coord, pt['coord'])
                                if d < min_err: name_matches = [idx]; min_err = d
                    e_indices = name_matches
                else:
                    if final_dest:
                        min_end_dist = 999999; best_e = 0
                        for idx, pt in enumerate(aw):
                            d = fast_dist_nm(final_dest, pt['coord'])
                            if d < min_end_dist: min_end_dist = d; best_e = idx
                        e_indices = [best_e]
                
                if e_indices:
                    best_path = None; min_path_len = 9999999
                    for s in s_indices:
                        for e in e_indices:
                            direction = 1 if s <= e else -1
                            dist = abs(e - s) 
                            if dist < min_path_len: min_path_len = dist; best_path = (s, e, direction)
                    if best_path:
                        s, e, direction = best_path
                        if direction == 1: raw = aw[s:e+1]
                        else: raw = aw[e:s+1][::-1]
                        
                        if raw and is_valid_coord(prev_coord) and is_valid_coord(raw[0]['coord']):
                            if fast_dist_nm(prev_coord, raw[0]['coord']) < 2: raw = raw[1:]
                        
                        current_valid_pos = prev_coord
                        for j, pt_data in enumerate(raw):
                            curr = pt_data['coord']
                            if not is_valid_coord(curr): continue
                            jump_dist = fast_dist_nm(current_valid_pos, curr)
                            if jump_dist > MAX_VALID_SEGMENT_NM: continue 
                            
                            for ip in interpolate_segment(current_valid_pos, curr, 50):
                                coords_info.append({'coord': ip, 'name': None, 'is_china': False})
                                
                            name = pt_data['name']
                            approx = (round(curr[0], 2), round(curr[1], 2))
                            is_cn = (name, approx) in self.china_nodes
                            coords_info.append({'coord': curr, 'name': name, 'is_china': is_cn})
                            
                            current_valid_pos = curr
                        if coords_info: prev_coord = coords_info[-1]['coord']
            else:
                sel = None
                if get_airport_coords(t): sel = get_airport_coords(t)
                elif t in self.global_db:
                    sel = min(self.global_db[t], key=lambda p: fast_dist_nm(p, final_dest) if final_dest else 0)
                if sel and is_valid_coord(sel):
                    if prev_coord and is_valid_coord(prev_coord):
                        jump_dist = fast_dist_nm(prev_coord, sel)
                        if jump_dist <= MAX_VALID_SEGMENT_NM:
                            for ip in interpolate_segment(prev_coord, sel, 50):
                                coords_info.append({'coord': ip, 'name': None, 'is_china': False})
                            approx = (round(sel[0], 2), round(sel[1], 2))
                            is_cn = (t, approx) in self.china_nodes
                            coords_info.append({'coord': sel, 'name': t, 'is_china': is_cn})
                            prev_coord = sel
                    else:
                        approx = (round(sel[0], 2), round(sel[1], 2))
                        is_cn = (t, approx) in self.china_nodes
                        coords_info.append({'coord': sel, 'name': t, 'is_china': is_cn})
                        prev_coord = sel
                        
        return coords_info

# ---------------------------------------------------------
# UI 메인 블록
# ---------------------------------------------------------
st.title("🌪️ Typhoon Flight Analyzer (Precision Route Engine)")

wp_df, aw_df, route_df, fix_df = load_static_db()
if wp_df is None or aw_df is None or route_df is None:
    st.error("🚨 필수 DB 파일이 누락되었습니다. (`Waypoint.xlsx`, `airway.xlsx`, `DB_ROUTE.xlsx`)")
    st.stop()
else:
    if 'engine' not in st.session_state:
        with st.spinner("📦 듀얼 코어 및 정밀 라우팅 모델 초기화 중..."):
            st.session_state.engine = DualCoreEngine(wp_df, aw_df, route_df, fix_df)
            st.session_state.engine.build_db()
        st.success(f"✅ 엔진 로드 완료 (총 {len(st.session_state.engine.global_db):,}개의 웨이포인트 장착!)")

col1, col2 = st.columns(2)
with col1: f_skd = st.file_uploader("✈️ SKD_BASE 업로드", type=['xlsx'])
with col2: f_rest = st.file_uploader("🌪️ Restrictions 업로드", type=['xlsx'])

if f_skd and f_rest:
    if st.button("🚀 정밀 비행편 분석 시작", type="primary", use_container_width=True):
        with st.spinner("태풍 회피 및 영공 통과 시간을 정밀 분석 중입니다..."):
            eng = st.session_state.engine
            skd_df = pd.read_excel(f_skd)
            rest_df = pd.read_excel(f_rest)
            
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
                    f_no = str(row.iloc[1]); dep = str(row.iloc[8]).strip(); arr = str(row.iloc[9]).strip()
                    status_text.text(f"분석 중: {f_no} ({dep}->{arr})")
                    
                    dep_keys, arr_keys = get_codes(dep), get_codes(arr)
                    mask_d = eng.db_route_df.iloc[:, 0].astype(str).str.upper().isin(dep_keys)
                    mask_a = eng.db_route_df.iloc[:, 1].astype(str).str.upper().isin(arr_keys)
                    matched_routes = eng.db_route_df[mask_d & mask_a]
                    
                    if matched_routes.empty: continue
                    d_raw = pd.to_datetime(row.iloc[2], errors='coerce')
                    if pd.isna(d_raw): continue
                    
                    def _t(x): 
                        if pd.isna(x): return None
                        if isinstance(x, time): return x
                        if isinstance(x, datetime): return x.time()
                        s = str(x).strip()
                        if len(s)==4 and s.isdigit(): return time(int(s[:2]), int(s[2:]))
                        if ':' in s: 
                            try: return pd.to_datetime(s).time()
                            except: return None
                        return None

                    t_std = _t(row.iloc[10]); t_sta = _t(row.iloc[11])
                    if not t_std or not t_sta: continue

                    dt_std = datetime.combine(d_raw.date(), t_std)
                    dt_sta = datetime.combine(d_raw.date(), t_sta)
                    if dt_sta < dt_std: dt_sta += timedelta(days=1)
                    fly_hours = (dt_sta - dt_std).total_seconds() / 3600 - 0.5
                    if fly_hours < 0.5: fly_hours = 0.5

                    route_objs = []
                    for _, r in matched_routes.iterrows():
                        r_name = str(r.iloc[2]).strip().upper()
                        if not (r_name.startswith('P') or r_name.startswith('W')): continue
                        r_data = eng.get_route_data(r_name, str(r.iloc[4]), dep, arr)
                        if r_data: route_objs.append({'name': r_name, 'data': r_data})
                        
                    if not route_objs: continue
                    
                    ref_route = next((r for r in route_objs if 'P01' in r['name'].upper()), None)
                    if not ref_route: ref_route = min(route_objs, key=lambda x: x['data']['total_dist'])
                    avg_speed = ref_route['data']['total_dist'] / fly_hours
                    if avg_speed < 100: avg_speed = 450.0 
                    
                    risk_routes = []
                    safe_list = []
                    china_transit_list = []
                    
                    for r in route_objs:
                        r_name = r['name']; r_data = r['data']
                        est_hours = r_data['total_dist'] / avg_speed
                        est_fly_time = timedelta(hours=est_hours)
                        
                        hit = False; hit_msg = ""
                        china_pts = [] 
                        
                        for k, info in enumerate(r_data['info']):
                            progress = r_data['seg_dists'][k] / r_data['total_dist'] if r_data['total_dist'] > 0 else 0
                            p_time = dt_std + timedelta(minutes=15) + (est_fly_time * progress)
                            
                            if info['is_china']:
                                china_pts.append((info['name'], p_time))
                                
                            if not hit:
                                for ty in typhoons:
                                    if fast_dist_nm(info['coord'], ty['c']) <= ty['r']:
                                        if ty['s'] <= p_time <= ty['e']:
                                            hit = True; hit_msg = f"{r_name}({ty['n']})"
                                            break
                        
                        if hit: risk_routes.append(hit_msg)
                        else: safe_list.append({'name': r_name, 'dist': r_data['total_dist']})
                        
                        if china_pts:
                            entry = china_pts[0]
                            exit_ = china_pts[-1]
                            if entry[0] == exit_[0]:
                                china_transit_list.append(f"{r_name} ({entry[0]} {entry[1].strftime('%H:%M')})")
                            else:
                                china_transit_list.append(f"{r_name} ({entry[0]} {entry[1].strftime('%H:%M')} ~ {exit_[1].strftime('%H:%M')} {exit_[0]})")
                    
                    if risk_routes:
                        safe_list.sort(key=lambda x: x['dist'])
                        res_list.append({
                            'FLT': f_no, 'DATE': str(d_raw.date()), 'DEP': dep, 'ARR': arr,
                            'STD': t_std.strftime("%H:%M"), 'STA': t_sta.strftime("%H:%M"),
                            'RESTRICTED_ROUTES': ", ".join(risk_routes),
                            'CHINA_TRANSIT': ", ".join(china_transit_list) if china_transit_list else "N/A",
                            'REC_ROUTE_1': safe_list[0]['name'] if len(safe_list)>0 else "N/A",
                            'DIST_1': f"{safe_list[0]['dist']:.0f}" if len(safe_list)>0 else "",
                            'REC_ROUTE_2': safe_list[1]['name'] if len(safe_list)>1 else "",
                            'DIST_2': f"{safe_list[1]['dist']:.0f}" if len(safe_list)>1 else "",
                            'REC_ROUTE_3': safe_list[2]['name'] if len(safe_list)>2 else "",
                            'DIST_3': f"{safe_list[2]['dist']:.0f}" if len(safe_list)>2 else "",
                            'REC_ROUTE_4': safe_list[3]['name'] if len(safe_list)>3 else "",
                            'DIST_4': f"{safe_list[3]['dist']:.0f}" if len(safe_list)>3 else ""
                        })
                except Exception as e: 
                    continue

            status_text.empty()
            if res_list:
                df_res = pd.DataFrame(res_list)
                st.success(f"🔥 총 {len(df_res)}건의 제한 운항편이 발견되었습니다!")
                st.dataframe(df_res)
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_res.to_excel(writer, index=False, sheet_name='Summary')
                    for d in df_res['DATE'].unique():
                        sub_df = df_res[df_res['DATE'] == d]
                        sub_df.to_excel(writer, index=False, sheet_name=f"RES_{d}")
                
                st.download_button(label="💾 최종 분석 결과 엑셀 다운로드", data=output.getvalue(), file_name="Typhoon_Analysis_Result.xlsx", mime="application/vnd.ms-excel")
            else:
                st.success("✅ 태풍의 영향을 받는 제한 운항편이 없습니다.")
