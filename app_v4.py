"""
가품 탐지 Streamlit 앱 v4
========================================
피처: price_rate_num, lap_v3_gap, store_rank, store_like_log,
      label_dist (DINOv2), clip_dist (CLIP) + platform OHE
모델: CatBoost Stage2 (model_cat_v2.pkl)
공식 이미지: official_product_dataset.xlsx 기준
조말론: 브랜드+용량 평균 임베딩 사용

실행: streamlit run app_v4.py
필요 파일: model_cat_v2.pkl, official_product_dataset.xlsx
"""

import re
import pickle
import warnings
import numpy as np
import pandas as pd
import cv2
import requests
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')

from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, AutoModel

# ════════════════════════════════════════════════════════════
# 0. 설정
# ════════════════════════════════════════════════════════════
MODEL_PATH   = 'model_cat_v2.pkl'
OFFICIAL_PATH = 'official_product_dataset.xlsx'
EMBED_PATH   = 'official_embeddings_v2.pkl'
THRESHOLD    = 0.4

PLATFORM_COLS = ['platform_11번가', 'platform_Gmarket', 'platform_lotteon',
                 'platform_smartstore', 'platform_ssg']
FEATURES = ['price_rate_num', 'store_rank', 'store_like_log'] + PLATFORM_COLS

FEATURE_LABELS = {
    'price_rate_num':      '공식가 대비 가격비율',
    'store_rank':          '판매자 랭크',
    'store_like_log':      '판매자 찜 수 (log)',
    'platform_11번가':     '플랫폼_11번가',
    'platform_Gmarket':    '플랫폼_Gmarket',
    'platform_lotteon':    '플랫폼_롯데온',
    'platform_smartstore': '플랫폼_스마트스토어',
    'platform_ssg':        '플랫폼_SSG',
}

_device = torch.device('cpu')

# ════════════════════════════════════════════════════════════
# 1. 모델 & 리소스 로드 (캐시)
# ════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_clip():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
    model.eval()
    return model, processor

@st.cache_resource
def load_dino():
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(_device)
    model.eval()
    return model, processor

@st.cache_resource
def load_official_data():
    """official_embeddings_v2.pkl 로드 (precompute_official_embeddings.py로 사전 생성 필요)"""
    import os
    if not os.path.exists(EMBED_PATH):
        st.error(f"❌ {EMBED_PATH} 파일이 없어요! precompute_official_embeddings.py 먼저 실행해주세요.")
        st.stop()
    with open(EMBED_PATH, 'rb') as f:
        return pickle.load(f)

# ════════════════════════════════════════════════════════════
# 2. 이미지 처리 함수
# ════════════════════════════════════════════════════════════
# ── 도메인별 헤더 설정 ──
_HEADERS_DEFAULT = {
    'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}
_HEADERS_BY_DOMAIN = {
    'jomalone.co.kr':    {**_HEADERS_DEFAULT, 'Referer': 'https://www.jomalone.co.kr/'},
    'esteelauder.co.kr': {**_HEADERS_DEFAULT, 'Referer': 'https://www.esteelauder.co.kr/'},
    'kiehls.co.kr':      {**_HEADERS_DEFAULT, 'Referer': 'https://www.kiehls.co.kr/'},
    '011st.com':         {**_HEADERS_DEFAULT, 'Referer': 'https://www.11st.co.kr/'},
    'ssgcdn.com':        {**_HEADERS_DEFAULT, 'Referer': 'https://www.ssg.com/'},
    'ssgdfs.com':        {**_HEADERS_DEFAULT, 'Referer': 'https://www.ssgdfs.com/'},
    'lotteon':           {**_HEADERS_DEFAULT, 'Referer': 'https://www.lotteon.com/'},
    'naver':             {**_HEADERS_DEFAULT, 'Referer': 'https://smartstore.naver.com/'},
    'gstatic':           {**_HEADERS_DEFAULT, 'Referer': 'https://www.gmarket.co.kr/'},
}

def _get_headers(url):
    for domain, h in _HEADERS_BY_DOMAIN.items():
        if domain in url:
            return h
    return _HEADERS_DEFAULT

def download_pil(url, timeout=10, max_retries=2):
    if not isinstance(url, str) or not url.strip():
        return None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=_get_headers(url), timeout=timeout)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            if attempt < max_retries:
                import time; time.sleep(1)
    return None

def get_clip_emb(img_pil, model, processor):
    inputs = processor(images=img_pil, return_tensors="pt").to(_device)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
    if not isinstance(feat, torch.Tensor):
        feat = feat.pooler_output
    return F.normalize(feat.float(), dim=-1).cpu()

def get_dino_emb(img_pil, model, processor):
    inputs = processor(images=img_pil, return_tensors="pt").to(_device)
    with torch.no_grad():
        out  = model(**inputs)
        feat = out.last_hidden_state[:, 0]
    return F.normalize(feat.float(), dim=-1).cpu()

def cosine_dist(a, b):
    return float(1 - F.cosine_similarity(a, b).item())

def calc_laplacian(img_pil):
    if img_pil is None:
        return 0.0
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# ════════════════════════════════════════════════════════════
# 3. 플랫폼별 크롤러
# ════════════════════════════════════════════════════════════
def crawl_11st(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'ko-KR,ko;q=0.9',
            'Referer': 'https://www.11st.co.kr/',
        }
        resp = requests.get(url, headers=headers, timeout=15)
        html = resp.text
        name_m  = re.search(r'"name"\s*:\s*"([^"]+)"', html)
        price_m = re.search(r'"price"\s*:\s*(\d+)', html)
        img_m   = re.search(r'"image"\s*:\s*"([^"]+)"', html)
        return {
            'product_name':      name_m.group(1) if name_m else '',
            'price':             int(price_m.group(1)) if price_m else None,
            'image_url':         img_m.group(1) if img_m else None,
            'delivery_overseas': 1 if '해외배송' in html else 0,
            'platform':          '11번가',
        }
    except Exception:
        return None

def crawl_ssg(url):
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True,
                args=['--no-sandbox','--disable-dev-shm-usage','--disable-gpu'])
            ctx  = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
                locale='ko-KR')
            page = ctx.new_page()
            page.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
            page.goto(url, wait_until='networkidle', timeout=40000)
            page.wait_for_timeout(3000)

            try:    name = page.locator('span.cdtl_info_tit_txt').first.inner_text().strip()
            except: name = ''

            prices = page.evaluate(r"""() => {
                const sels=['.ssg_price.notranslate','.ssg_price','[class*="price"]'];
                for(const s of sels){
                    const vals=Array.from(document.querySelectorAll(s))
                        .map(e=>e.innerText.trim()).filter(t=>t.match(/^\d[\d,]+$/));
                    if(vals.length>0)return vals;
                }return[];
            }""")
            price = int(re.sub(r'[^\d]', '', prices[0])) if prices else None

            img_url = page.evaluate(r"""() => {
                const n=src=>(!src?'':src.startsWith('//')?'https:'+src:src);
                for(const img of document.querySelectorAll('img')){
                    const s=n(img.src||'');
                    if(s&&s.includes('sitem.ssgcdn'))return s;
                }return '';
            }""")

            del_text = page.evaluate(r"""() => {
                const el=document.querySelector('.cdtl_delivery_info,[class*="delivery_wrap"]');
                return el?el.innerText:'';
            }""")
            browser.close()

        return {
            'product_name': name, 'price': price,
            'image_url': img_url,
            'delivery_overseas': 1 if '해외' in del_text else 0,
            'platform': 'ssg',
        }
    except Exception as e:
        st.warning(f"SSG 크롤링 오류: {e}")
        return None

def crawl_lotteon(url):
    try:
        import json, ast
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        opts = Options()
        opts.add_argument('--headless')
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        opts.add_argument('user-agent=Mozilla/5.0')
        driver = webdriver.Chrome(options=opts)
        driver.get(url)
        wait = WebDriverWait(driver, 10)

        try:
            name = wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, '.pd-widget1__product-name'))).text.strip()
        except: name = ''

        try:
            price_text = driver.execute_script("""
                const els=Array.from(document.querySelectorAll('[class*="price"]'));
                const el=els.find(e=>e.innerText.includes('판매가'));
                if(!el)return'';
                const m=el.innerText.match(/판매가[\\s\\n]+([\\d,]+)/);
                return m?m[1]:'';
            """)
            price = int(re.sub(r'[^\d]', '', price_text)) if price_text else None
        except: price = None

        try:
            img_url = driver.execute_script("""
                return Array.from(document.querySelectorAll('img'))
                    .map(i=>i.src||i.dataset?.src||'')
                    .find(s=>s.includes('contents.lotteon.com/itemimage'))||'';
            """)
        except: img_url = None

        try:
            del_text = driver.find_element(
                By.XPATH,
                '//*[@id="stickyOptionParent"]/div/div/div/div[1]/div/div/div[1]/dl[1]/dd/p[1]'
            ).text
        except: del_text = ''

        store_rank = np.nan
        try:
            meta = driver.find_element(By.ID, 'metaData').get_attribute('value')
            try:    meta_json = json.loads(meta)
            except: meta_json = ast.literal_eval(meta)
            grade = meta_json.get('product',{}).get('basicInfo',{}).get('sellerGrade','')
            raw = {'SUPER':3,'PREMIUM':2,'STANDARD':1,'BASIC':0}.get(str(grade).upper(), np.nan)
            store_rank = raw / 3 if not pd.isna(raw) else np.nan
        except: pass

        driver.quit()
        return {
            'product_name': name, 'price': price,
            'image_url': img_url,
            'delivery_overseas': 1 if '해외' in del_text else 0,
            'store_rank': store_rank,
            'platform': 'lotteon',
        }
    except Exception as e:
        st.warning(f"롯데온 크롤링 오류: {e}")
        return None

# ════════════════════════════════════════════════════════════
# 4. 피처 빌드
# ════════════════════════════════════════════════════════════
def build_features(info, official_map):
    f = {}
    brand    = info['brand_name']
    volume   = int(info['volume'])
    platform = info['platform']
    price    = info['price']

    key = (brand, int(float(volume)))
    off = official_map.get(key)

    # price_rate_num (수량 고려: 판매가 / (공식가 * 수량))
    official_price = off['price'] if off else None
    qty = int(info.get('quantity', 1) or 1)
    f['price_rate_num'] = round(price / (official_price * qty), 4) if (official_price and price) else np.nan

    # store_like_log
    f['store_like_log'] = float(np.log1p(info.get('store_like_count', 0) or 0))

    # store_rank
    f['store_rank'] = info.get('store_rank', np.nan)

    # platform OHE
    for p in ['11번가', 'Gmarket', 'lotteon', 'smartstore', 'ssg']:
        f[f'platform_{p}'] = 1.0 if platform == p else 0.0

    return pd.DataFrame([f])[FEATURES]

# ════════════════════════════════════════════════════════════
# 5. Streamlit UI
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="가품 탐지기 v4", page_icon="🔍", layout="centered")
st.title("🔍 화장품 가품 탐지기 v4")
st.caption("상품 URL 입력 → 자동 크롤링 → CLIP/DINOv2 이미지 분석 → 가품 확률 예측")

# 모델 & 공식 데이터 로드
try:
    model = load_model()
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

official_map = load_official_data()
st.success(f"✅ 공식 데이터 로드 완료 ({len(official_map)}개 브랜드+용량)")

# ── 입력 폼 ──────────────────────────────────────────────
with st.form("input_form"):
    platform_ui = st.selectbox("플랫폼 선택", ["11번가", "SSG", "롯데온"])
    url = st.text_input("상품 URL", placeholder="해당 플랫폼 상품 URL을 붙여넣으세요")

    col1, col2 = st.columns(2)
    with col1:
        brand    = st.selectbox("브랜드", ["Jo_Malone", "Kiehls", "Estee_Lauder"])
        volume   = st.number_input("용량 (ml)", min_value=0, value=100, step=1)
        quantity = st.number_input("수량 (개)", min_value=1, value=1, step=1)
    with col2:
        store_like = st.number_input("판매자 찜 수", min_value=0, value=0, step=1)
        if platform_ui == "11번가":
            rank_label = st.selectbox("판매자 랭크 (숫자 클수록 높음)",
                ["없음/모름", "0", "1", "2", "3", "4", "5"])
            store_rank = np.nan if rank_label == "없음/모름" else int(rank_label) / 5
        elif platform_ui == "SSG":
            rank_label = st.selectbox("판매자 랭크 (숫자 클수록 높음)",
                ["없음/모름", "0", "1", "2", "3", "4"])
            store_rank = np.nan if rank_label == "없음/모름" else int(rank_label) / 4
        else:
            st.info("롯데온 판매자 랭크는 페이지에서 자동 추출됩니다.")
            store_rank = np.nan

    submitted = st.form_submit_button("🔍 분석 시작", use_container_width=True)

# ── 분석 실행 ────────────────────────────────────────────
if submitted and url:
    platform_key = {'11번가': '11번가', 'SSG': 'ssg', '롯데온': 'lotteon'}[platform_ui]

    with st.spinner(f"{platform_ui} 상품 크롤링 중..."):
        product = {'11번가': crawl_11st, 'SSG': crawl_ssg, '롯데온': crawl_lotteon}[platform_ui](url)

    if product is None:
        st.error("상품 정보를 가져오지 못했어요. URL을 확인해주세요.")
        st.stop()

    # 상품 정보 표시
    st.divider()
    st.subheader("📦 상품 정보")
    c1, c2 = st.columns([1, 2])
    with c1:
        if product.get('image_url'):
            st.image(product['image_url'], width=200)
    with c2:
        st.write(f"**상품명:** {product.get('product_name', '-')}")
        key = (brand, int(volume))
        off = official_map.get(key)
        official_price = off['price'] if off else None
        if product.get('price') and official_price:
            price_rate = product['price'] / (official_price * int(quantity))
            st.write(f"**가격:** {product['price']:,}원")
            st.write(f"**공식가:** {official_price:,}원 × {int(quantity)}개 = {official_price * int(quantity):,}원")
            st.write(f"**공식가 대비:** {price_rate:.1%}")
        elif product.get('price'):
            st.write(f"**가격:** {product['price']:,}원")
            st.warning(f"⚠️ ({brand}, {int(volume)}ml) 공식가 정보 없음")
        st.write(f"**해외배송:** {'예' if product.get('delivery_overseas') else '아니오'}")

    # 피처 빌드
    with st.spinner("피처 계산 중..."):
        info = {
            'brand_name':       brand,
            'volume':           volume,
            'quantity':         quantity,
            'platform':         platform_key,
            'price':            product.get('price'),
            'image_url':        product.get('image_url'),
            'store_like_count': store_like,
            'store_rank':       product.get('store_rank', store_rank),
        }
        X = build_features(info, official_map)

    # 예측
    prob = float(model.predict_proba(X)[0][1])

    # 결과 표시
    st.divider()
    st.subheader("🎯 분석 결과")
    if prob >= 0.5:
        st.error("🔴 **가품 의심**")
    else:
        st.success("🟢 **정품 가능성 높음**")

    # SHAP 분석
    with st.expander("📊 판정 근거 (SHAP 분석)", expanded=True):
        try:
            import shap
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.rcParams['font.family'] = ['AppleGothic', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False

            explainer = shap.TreeExplainer(model)
            shap_raw  = explainer.shap_values(X.astype(float))

            # CatBoost 2D ndarray 처리
            if isinstance(shap_raw, list):
                shap_arr = shap_raw[1][0]
            elif shap_raw.ndim == 3:
                shap_arr = shap_raw[0, :, 1]
            else:
                shap_arr = shap_raw[0]

            shap_df = pd.DataFrame({
                '피처':   [FEATURE_LABELS.get(f, f) for f in FEATURES],
                '피처값': [f"{v:.4f}" if pd.notna(v) else "NaN" for v in X.iloc[0].values],
                'SHAP_float': shap_arr,
            })
            shap_df['영향'] = shap_df['SHAP_float'].apply(
                lambda x: '🔴 가품 방향' if x > 0.001
                else ('🟢 정품 방향' if x < -0.001 else '➖ 중립'))
            shap_df = shap_df.sort_values('SHAP_float', key=abs, ascending=False)
            shap_df['SHAP값'] = shap_df['SHAP_float'].apply(lambda x: f"{x:+.4f}")
            st.dataframe(
                shap_df[['피처', '피처값', 'SHAP값', '영향']].reset_index(drop=True),
                use_container_width=True)

            st.write("**SHAP 워터폴 차트**")
            explanation = shap.Explanation(
                values=shap_arr,
                base_values=float(explainer.expected_value),
                data=X.astype(float).iloc[0].values,
                feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURES]
            )
            fig, _ = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(explanation, max_display=10, show=False)
            st.pyplot(fig)
            plt.close()

        except ImportError:
            st.info("SHAP 설치 필요: pip install shap")
        except Exception as e:
            st.error(f"SHAP 분석 오류: {e}")

    with st.expander("📋 피처 원본값"):
        display_df = X.T.rename(columns={0: '값'})
        display_df.index = [FEATURE_LABELS.get(i, i) for i in display_df.index]
        display_df['값'] = display_df['값'].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
        st.dataframe(display_df, use_container_width=True)

elif submitted and not url:
    st.warning("URL을 입력해주세요.")
