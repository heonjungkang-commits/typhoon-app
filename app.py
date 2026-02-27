# ==========================================
# [Final v38.1] 태풍 분석 시스템 (CHN/SEA 전체 표출 + 수동 입력 허용)
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from datetime import datetime, timedelta, time
import airportsdata
import io
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------
# [STREAMLIT CONFIG & CUSTOM CSS]
# ---------------------------------------------------------
st.set_page_config(page_title="Typhoon Flight Analyzer", layout="wide", page_icon="✈️")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #1E3A8A; font-weight: 700; margin-bottom: 1rem; }
    h2, h3 { color: #2563EB; font-weight: 600; margin-top: 1rem; }
    .stButton>button { border-radius: 8px; font-weight: bold; height: 3rem; }
    th { background-color: #F3F4F6 !important; color: #111827 !important; }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3211/3211184.png", width=60)
    st.header("⚙️ 시스템 설정")
    USE_INTERPOLATION = st.checkbox("내삽(Interpolation) 정밀 연산", value=True)
    MAX_VALID_SEGMENT_NM = st.number_input("점프 방지 거리(nm)", value=600, step=50)
    st.markdown("---")
    st.info("💡 **엔진 상태:**\n- **CHN/SEA 바운드 전체 표출 [ON]**\n- 그 외 P-Route 제한편 표출 [ON]\n- 엑셀 동적 수식 연동 [ON]\n- 최종항로 수동입력 허용 [ON]")

# ---------------------------------------------------------
# 1. 고정 데이터 & 유틸리티
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_airports():
    return airportsdata.load('iata'), airportsdata.load('icao')

airports_iata, airports_icao = load_airports()

@st.cache_data(show_spinner=False)
def load_static_db():
    try:
        wp_raw = pd.read_excel("Waypoint.xlsx")
        aw_raw = pd.read_excel("airway.xlsx")
        rte_raw = pd.read_excel("DB_ROUTE.xlsx")
        
        try: city_pair_raw = pd.read_excel("CITY PAIR.xlsx")
        except: city_pair_raw = None
        
        try: sxx_raw = pd.read_excel("SXX.xlsx")
        except: sxx_raw = None
        
        try: fix_raw = pd.read_csv("FIX_result.csv")
        except: fix_raw = None
        
        wp_df = wp_raw.dropna(subset=[wp_raw.columns[0]])
        aw_df = aw_raw.dropna(subset=[aw_raw.columns[0]])
        route_df = rte_raw.dropna(how='all')
        
        city_pair_dict = {}
        if city_pair_raw is not None:
            for _, r in city_pair_raw.iterrows():
                try: city_pair_dict[str(r.iloc[0]).strip()] = str(r.iloc[1]).strip()
                except: pass
                
        sxx_dict = {'SO': {}, 'SI': {}}
        if sxx_raw is not None:
            for _, r in sxx_raw.iterrows():
                try:
                    code = str(r.iloc[0]).strip().upper()
                    apt = str(r.iloc[1]).strip().upper()
                    if code.startswith('SO'): sxx_dict['SO'][apt] = code
                    elif code.startswith('SI'): sxx_dict['SI'][apt] = code
                except: pass
                
        return wp_df, aw_df, route_df, fix_raw, city_pair_dict, sxx_dict
    except FileNotFoundError:
        return None, None, None, None, {}, {'SO': {}, 'SI': {}}

def parse_wkt_point(wkt_str):
    try:
        match = re.match(r'POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)', str(wkt_str).upper())
        if match: return (float(match.group(2)), float(match.group(1)))
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

def get_s_xx(dep, arr, sxx_dict):
    d_iata = next((k for k in get_codes(dep) if len(k)==3), dep[:3])
    a_iata = next((k for k in get_codes(arr) if len(k)==3), arr[:3])

    if a_iata in sxx_dict['SO']: return sxx_dict['SO'][a_iata]
    if d_iata in sxx_dict['SI']: return sxx_dict['SI'][d_iata]
    return ''

# ---------------------------------------------------------
# 2. 듀얼 코어 & 엔진
# ---------------------------------------------------------
class DualCoreEngine:
    def __init__(self, wp_df, aw_df, route_df, fix_df):
        self.wp_df = wp_df; self.airway_df = aw_df; self.db_route_df = route_df; self.fix_df = fix_df
        self.global_db = {}; self.airway_dict = {}; self.route_cache = {}
        self.china_nodes = set()

    def build_db(self):
        seen_coords = set()
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
                    if fir.startswith('Z') and not fir.startswith('ZK') and not fir.startswith('ZM') and not fir.startswith('ZJ'):
                        self.china_nodes.add((n, approx))

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
                if cc.startswith('Z') and not cc.startswith('ZK') and not cc.startswith('ZM') and not cc.startswith('ZJ'):
                    self.china_nodes.add((n, approx))
        
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
            is_cn = any(len(k)==4 and k.startswith('Z') and not k.startswith('ZK') and not k.startswith('ZM') and not k.startswith('ZJ') for k in dep_keys)
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
st.title("Typhoon Flight Analyzer")
st.markdown("항공기 운항 스케줄과 태풍 데이터를 교차 분석하여 제한 운항편 및 최적 우회 항로를 도출합니다.")

wp_df, aw_df, route_df, fix_df, city_pair_dict, sxx_dict = load_static_db()

if wp_df is None or aw_df is None or route_df is None:
    st.error("🚨 필수 DB 파일이 누락되었습니다. (`Waypoint.xlsx`, `airway.xlsx`, `DB_ROUTE.xlsx`)")
    st.stop()
else:
    if 'engine' not in st.session_state:
        with st.spinner("📦 듀얼 코어 엔진 초기화 중..."):
            st.session_state.engine = DualCoreEngine(wp_df, aw_df, route_df, fix_df)
            st.session_state.engine.build_db()

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("🛫 1. 스케줄 데이터 업로드")
    f_skd = st.file_uploader("SKD_BASE (CSV 포맷)", type=['csv'], label_visibility="collapsed")
    if f_skd:
        st.success("스케줄 파일 업로드 완료")

with col_right:
    st.subheader("🌪️ 2. 태풍 데이터 입력")
    st.caption("표를 클릭하여 직접 수정/추가할 수 있습니다.")
    
    if 'typhoon_input_data' not in st.session_state:
        st.session_state.typhoon_input_data = pd.DataFrame({
            '태풍명': ['HINNAMNOR', '', ''],
            '위도(Lat)': [25.5, None, None],
            '경도(Lon)': [125.5, None, None],
            '시작일시': [datetime.now().strftime("%Y-%m-%d %H:%M"), '', ''],
            '종료일시': [(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M"), '', ''],
            '반경(nm)': [300.0, None, None]
        })

    edited_typhoons = st.data_editor(
        st.session_state.typhoon_input_data, 
        num_rows="dynamic", 
        use_container_width=True, 
        height=210
    )
    st.session_state.typhoon_input_data = edited_typhoons

if f_skd:
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
        st.session_state.df_res = None
        st.session_state.excel_data = None
        st.session_state.map_store = {}
        st.session_state.typhoons = []
        st.session_state.total_skd_len = 0

    if st.button("🚀 정밀 비행편 분석 시작", type="primary", use_container_width=True):
        st.session_state.analysis_done = False 
        
        with st.status("🔍 정밀 비행편 분석 진행 중...", expanded=True) as status:
            eng = st.session_state.engine
            
            st.write("1. 스케줄 CSV 로드 및 인코딩 처리 중...")
            try:
                skd_df = pd.read_csv(f_skd, encoding='utf-8-sig')
            except UnicodeDecodeError:
                f_skd.seek(0)
                skd_df = pd.read_csv(f_skd, encoding='cp949') 
            except Exception as e:
                status.update(label="에러 발생", state="error", expanded=False)
                st.error("CSV 파일을 읽는 중 오류가 발생했습니다.")
                st.stop()
                
            st.session_state.total_skd_len = len(skd_df)
            
            typhoons = []
            for _, r in edited_typhoons.iterrows():
                try:
                    if pd.isna(r['위도(Lat)']) or pd.isna(r['경도(Lon)']) or not str(r['태풍명']).strip(): continue
                    typhoons.append({
                        'n': str(r['태풍명']).strip(),
                        'c': (float(r['위도(Lat)']), float(r['경도(Lon)'])),
                        's': pd.to_datetime(r['시작일시']),
                        'e': pd.to_datetime(r['종료일시']),
                        'r': float(r['반경(nm)'])
                    })
                except: continue
            
            res_list = []
            map_store = {}
            progress_bar = st.progress(0)
            
            st.write("2. 운항편 항로 데이터 추출 및 태풍 반경 교차 검증 중...")
            for idx, row in skd_df.iterrows():
                progress_bar.progress((idx + 1) / len(skd_df))
                try:
                    f_no = str(row.iloc[1]); dep = str(row.iloc[8]).strip(); arr = str(row.iloc[9]).strip()
                    ac_type = str(row.iloc[5]).strip()
                    
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
                        
                        if not (r_name.startswith('P') or r_name.startswith('W') or r_name.startswith('T')): 
                            continue
                            
                        r_data = eng.get_route_data(r_name, str(r.iloc[4]), dep, arr)
                        if r_data: route_objs.append({'name': r_name, 'data': r_data})
                        
                    if not route_objs: continue
                    
                    ref_route = next((r for r in route_objs if 'P01' in r['name'].upper()), None)
                    if not ref_route: ref_route = min(route_objs, key=lambda x: x['data']['total_dist'])
                    avg_speed = ref_route['data']['total_dist'] / fly_hours
                    if avg_speed < 100: avg_speed = 450.0 
                    
                    p01_fly_mins = (ref_route['data']['total_dist'] / avg_speed) * 60
                    
                    risk_routes = []
                    safe_list = []
                    china_transit_list = [] # 중국 통과 정보 저장 (숨김 컬럼용)
                    
                    for r in route_objs:
                        r_name = r['name']; r_data = r['data']
                        est_hours = r_data['total_dist'] / avg_speed
                        est_fly_time = timedelta(hours=est_hours)
                        est_mins = est_hours * 60
                        
                        ft_increase = round(est_mins - p01_fly_mins) if r_name != ref_route['name'] else 0
                        if ft_increase < 0: ft_increase = 0 
                        
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
                        else: safe_list.append({'name': r_name, 'ft_inc': ft_increase})
                        
                        # 제한된 항로든 안전한 항로든 중국 통과 정보는 모두 수집해둠 (나중에 수동 검색 시 매칭을 위해)
                        if china_pts:
                            entry = china_pts[0]
                            exit_ = china_pts[-1]
                            if entry[0] == exit_[0]:
                                china_transit_list.append(f"{r_name} ({entry[0]} {entry[1].strftime('%H:%M')})")
                            else:
                                china_transit_list.append(f"{r_name} ({entry[0]} {entry[1].strftime('%H:%M')} ~ {exit_[1].strftime('%H:%M')} {exit_[0]})")
                    
                    # 🚨 [신규 필터 적용]: BND가 CHN, SEA 이면 무조건 표출, 아니면 P항로 제한 시에만 표출
                    has_p_risk = False
                    if risk_routes:
                        has_p_risk = any(r_str.startswith('P') for r_str in risk_routes)
                        
                    pair_key_1 = f"{dep[:3]}/{arr[:3]}"
                    pair_key_2 = f"{dep}/{arr}"
                    bound_val = city_pair_dict.get(pair_key_1, city_pair_dict.get(pair_key_2, ""))
                    
                    s_xx_val = get_s_xx(dep, arr, sxx_dict)
                    
                    # (CHN, SEA 바운드) 이거나 (그 외 바운드인데 P항로 위험이 있는 경우)
                    if bound_val in ['CHN', 'SEA'] or has_p_risk:
                        safe_list.sort(key=lambda x: x['ft_inc']) 
                        
                        res_list.append({
                            'BND': bound_val,
                            'DATE': str(d_raw.date()),
                            'FLT': f_no,
                            'FR': dep,
                            'TO': arr,
                            'STD': t_std.strftime("%H:%M"),
                            'STA': t_sta.strftime("%H:%M"),
                            'AC': ac_type, 
                            'C_RTE': s_xx_val,
                            # 예보시간 삭제됨
                            '항로목록': ", ".join(risk_routes) if risk_routes else "",
                            '항로명_1': safe_list[0]['name'] if len(safe_list)>0 else "N/A",
                            'F/T 증가_1': safe_list[0]['ft_inc'] if len(safe_list)>0 else "",
                            '항로명_2': safe_list[1]['name'] if len(safe_list)>1 else "",
                            'F/T 증가_2': safe_list[1]['ft_inc'] if len(safe_list)>1 else "",
                            '항로명_3': safe_list[2]['name'] if len(safe_list)>2 else "",
                            'F/T 증가_3': safe_list[2]['ft_inc'] if len(safe_list)>2 else "",
                            '항로명_4': safe_list[3]['name'] if len(safe_list)>3 else "",
                            'F/T 증가_4': safe_list[3]['ft_inc'] if len(safe_list)>3 else "",
                            '최종 사용항로': "",
                            '승무 구성': "",
                            '허가신청자': "",
                            '허가 필요 국가': "",
                            '중국통과우회항로': "", # Formula가 덮어쓸 자리
                            '허가 신청': "",
                            '허가 취득': "",
                            'Hidden_CHN_Info': "|".join(china_transit_list) if china_transit_list else "N/A"
                        })
                        
                        dep_c = get_airport_coords(dep_keys[0]) if dep_keys else None
                        arr_c = get_airport_coords(arr_keys[0]) if arr_keys else None
                        map_store[f"{f_no} ({dep}->{arr})"] = {
                            'dep_coord': dep_c, 'arr_coord': arr_c,
                            'routes': route_objs,
                            'risk_routes': risk_routes
                        }
                except Exception as e: 
                    continue

            st.session_state.map_store = map_store
            st.session_state.typhoons = typhoons
            
            st.write("3. 엑셀 동적 함수(Formula) 및 사내 폼 매핑 중...")
            if res_list:
                df_res = pd.DataFrame(res_list)
                
                # '예보시간'이 삭제되고 CHN Route Code -> 중국통과우회항로 로 변경된 최종 25개 컬럼 + 1히든 컬럼
                df_res.columns = [
                    'BND', 'DATE', 'FLT', 'FR', 'TO', 'STD', 'STA', 'AC', 'C_RTE', 
                    '항로목록', '항로명', 'F/T 증가', '항로명 ', 'F/T 증가 ', '항로명  ', 'F/T 증가  ', '항로명   ', 'F/T 증가   ', 
                    '최종 사용항로', '승무 구성', '허가신청자', '허가 필요 국가', '중국통과우회항로', '허가 신청', '허가 취득', 'Hidden_CHN_Info'
                ]
                
                st.session_state.df_res = df_res
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_res.to_excel(writer, index=False, sheet_name='Summary')
                    
                    def format_dynamic_excel(ws, df):
                        col_w_idx = df.columns.get_loc('중국통과우회항로') # 수식이 들어갈 22번 인덱스
                        col_z_idx = df.columns.get_loc('Hidden_CHN_Info') # 숨길 25번 인덱스
                        
                        ws.set_column(col_w_idx, col_w_idx, 35) # 넓이 확장
                        ws.set_column(col_z_idx, col_z_idx, None, None, {'hidden': 1}) # 데이터 숨김 처리
                        
                        for row_idx in range(len(df)):
                            excel_row = row_idx + 2
                            
                            # 🚨 [변경]: '최종 사용항로(S열)'의 드롭다운(Data Validation) 로직을 완전히 삭제하여 강제 수동 타이핑 허용!
                            
                            # '중국통과우회항로(W열)'에 엑셀 MID/SEARCH 함수 삽입
                            # 사용자가 최종 사용항로(S열)를 수동 입력하면 -> 숨겨진 Z열에서 해당 정보를 찾아내 표출시킴
                            formula = f'=IF(ISBLANK(S{excel_row}),"",IFERROR(MID(Z{excel_row},SEARCH(S{excel_row},Z{excel_row}),IFERROR(SEARCH("|",Z{excel_row},SEARCH(S{excel_row},Z{excel_row}))-SEARCH(S{excel_row},Z{excel_row}),LEN(Z{excel_row})-SEARCH(S{excel_row},Z{excel_row})+1)),"해당 항로 중국통과 정보 없음"))'
                            ws.write_formula(row_idx + 1, col_w_idx, formula)

                    format_dynamic_excel(writer.sheets['Summary'], df_res)
                    
                    for d in df_res['DATE'].unique():
                        sub_df = df_res[df_res['DATE'] == d]
                        sheet_name = f"RES_{d}"
                        sub_df.to_excel(writer, index=False, sheet_name=sheet_name)
                        format_dynamic_excel(writer.sheets[sheet_name], sub_df)
                
                st.session_state.excel_data = output.getvalue()
            else:
                st.session_state.df_res = None
                
            st.session_state.analysis_done = True
            
            status.update(label="✅ 데이터 분석 완료", state="complete", expanded=False)

    # ---------------------------------------------------------
    # 3. 대시보드 결과 표출 (Tabs 활용)
    # ---------------------------------------------------------
    if st.session_state.get('analysis_done'):
        st.markdown("---")
        st.subheader("💡 분석 요약 리포트")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("업로드된 총 스케줄", f"{st.session_state.total_skd_len:,}편")
        
        if st.session_state.df_res is not None:
            total_listed = len(st.session_state.df_res)
            # 🚨 [UI 업데이트] 전체 표출 개수 중 실제로 P항로가 제한된(항로목록에 P가 포함된) 개수를 카운트
            p_risk_count = st.session_state.df_res['항로목록'].str.contains(r'P\d', na=False, regex=True).sum()
            
            col_m2.metric("리포트 표출 운항편", f"{total_listed:,}편", delta=f"실제 제한편 {p_risk_count}편 포함", delta_color="inverse")
            col_m3.metric("안전성 상태", "주의 요망 ⚠️" if p_risk_count > 0 else "정상 운항 (모니터링) 🟢")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["📊 상세 분석 테이블", "🗺️ GIS 항로 시각화"])
            
            with tab1:
                st.download_button(
                    label="💾 엑셀 리포트 다운로드 (동적 수식 적용)", 
                    data=st.session_state.excel_data, 
                    file_name="Typhoon_Analysis_Result_Formatted.xlsx", 
                    mime="application/vnd.ms-excel"
                )
                # 웹 화면 표에서는 지저분한 히든 컬럼을 제거하고 깔끔하게 보여줍니다.
                st.dataframe(st.session_state.df_res.drop(columns=['Hidden_CHN_Info']), use_container_width=True, height=500)
                
            with tab2:
                if st.session_state.map_store:
                    flt_list = list(st.session_state.map_store.keys())
                    selected_flt = st.selectbox("지도를 확인할 운항편을 선택하세요:", ["선택하세요..."] + flt_list)
                    
                    if selected_flt != "선택하세요...":
                        m_data = st.session_state.map_store[selected_flt]
                        
                        if m_data['dep_coord'] and m_data['arr_coord']:
                            center_lat = (m_data['dep_coord'][0] + m_data['arr_coord'][0]) / 2
                            center_lon = (m_data['dep_coord'][1] + m_data['arr_coord'][1]) / 2
                        else:
                            center_lat, center_lon = 30.0, 125.0
                            
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
                        
                        for ty in st.session_state.typhoons:
                            folium.Circle(
                                location=ty['c'],
                                radius=ty['r'] * 1852,
                                color='red', weight=2, fill=True, fill_color='red', fill_opacity=0.3,
                                tooltip=f"태풍 {ty['n']} (반경 {ty['r']}nm)"
                            ).add_to(m)
                            
                        for r in m_data['routes']:
                            r_name = r['name']
                            coords = [pt['coord'] for pt in r['data']['info']]
                            is_risk = any(r_name in r_str for r_str in m_data['risk_routes'])
                            
                            color = 'red' if is_risk else '#2563EB'
                            weight = 4 if is_risk else 2
                            
                            folium.PolyLine(
                                locations=coords,
                                color=color,
                                weight=weight,
                                tooltip=f"{r_name} 항로 ({'위험 - 태풍 제한' if is_risk else '안전 - 우회 추천'})"
                            ).add_to(m)
                            
                        if m_data['dep_coord']:
                            folium.Marker(m_data['dep_coord'], popup="Departure", icon=folium.Icon(color='green', icon='plane')).add_to(m)
                        if m_data['arr_coord']:
                            folium.Marker(m_data['arr_coord'], popup="Arrival", icon=folium.Icon(color='blue', icon='flag')).add_to(m)
                            
                        st_folium(m, width=1400, height=650)
        else:
            col_m2.metric("리포트 표출 운항편", "0편", delta="ALL CLEAR", delta_color="normal")
            col_m3.metric("안전성 상태", "정상 운항 🟢")
            st.success("✅ 조건에 해당하는 표출 운항편이 없습니다.")
