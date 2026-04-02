from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -----------------------------
# 폰트 설정
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
FONT_PATH = BASE_DIR / "fonts" / "NotoSansKR-VariableFont_wght.ttf"

font_prop = None
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    font_prop = fm.FontProperties(fname=str(FONT_PATH))
    font_name = font_prop.get_name()
    plt.rcParams["font.family"] = font_name
else:
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="이커머스 반품 예측 대시보드",
    layout="wide"
)


# -----------------------------
# 파일 로드
# -----------------------------
@st.cache_data
def load_data():
    x_val = pd.read_csv(BASE_DIR / "X_val_unscaled.csv")
    y_val = pd.read_csv(BASE_DIR / "y_val.csv")
    val_pred = pd.read_csv(BASE_DIR / "val_predictions.csv")
    feat_imp = pd.read_csv(BASE_DIR / "feature_importance.csv")

    test_pred_path = BASE_DIR / "test_predictions.csv"
    test_pred = pd.read_csv(test_pred_path) if test_pred_path.exists() else None

    return x_val, y_val, val_pred, feat_imp, test_pred


# -----------------------------
# 원핫 컬럼 복원
# -----------------------------
def decode_onehot(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    if not cols:
        return pd.Series(["unknown"] * len(df), index=df.index)

    return df[cols].idxmax(axis=1).str.replace(prefix + "_", "", regex=False)


# -----------------------------
# 파생 변수 생성
# -----------------------------
def add_derived_columns(df):
    df = df.copy()

    df["product_category_label"] = decode_onehot(df, "product_category")
    df["device_type_label"] = decode_onehot(df, "device_type")
    df["shipping_method_label"] = decode_onehot(df, "shipping_method")
    df["payment_method_label"] = decode_onehot(df, "payment_method")

    # 배송 지연 관련 파생 변수
    df["is_delayed"] = (df["delivery_delay_days"] > 0).astype(int)
    df["delay_days_int"] = df["delivery_delay_days"].astype(int)

    df["discount_bin"] = pd.cut(
        df["discount_percent"],
        bins=[-0.1, 10, 30, 50, 100],
        labels=["0-10", "10-30", "30-50", "50+"]
    )

    df["delay_bin"] = pd.cut(
        df["delay_days_int"],
        bins=[-0.1, 1, 3, 100],
        labels=["0-1", "2-3", "4+"]
    )

    df["past_return_bin"] = pd.cut(
        df["past_return_rate"],
        bins=[-0.01, 0.2, 0.5, 1.0],
        labels=["Low", "Medium", "High"]
    )

    df["risk_level"] = pd.cut(
        df["pred_prob"],
        bins=[0.0, 0.3, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    return df


# -----------------------------
# 성능 계산
# -----------------------------
def calculate_metrics(df):
    if len(df) == 0:
        return 0, 0, 0, 0

    y_true = df["returned"]
    y_pred = df["pred_label"]

    accuracy = (y_true == y_pred).mean()

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, f1, precision, recall


# -----------------------------
# 한글 폰트 적용 보조 함수
# -----------------------------
def apply_korean_font(ax, title=None, xlabel=None, ylabel=None):
    if title is not None:
        if font_prop is not None:
            ax.set_title(title, fontproperties=font_prop)
        else:
            ax.set_title(title)

    if xlabel is not None:
        if font_prop is not None:
            ax.set_xlabel(xlabel, fontproperties=font_prop)
        else:
            ax.set_xlabel(xlabel)

    if ylabel is not None:
        if font_prop is not None:
            ax.set_ylabel(ylabel, fontproperties=font_prop)
        else:
            ax.set_ylabel(ylabel)

    if font_prop is not None:
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)


# -----------------------------
# 데이터 준비
# -----------------------------
try:
    X_val, y_val, val_pred, feature_importance, test_pred = load_data()
except Exception as e:
    st.error(f"파일 로드 중 오류가 발생했습니다: {e}")
    st.stop()

required_pred_cols = {"pred_prob", "pred_label"}
if not required_pred_cols.issubset(val_pred.columns):
    st.error("val_predictions.csv에는 pred_prob, pred_label 컬럼이 있어야 합니다.")
    st.stop()

if "returned" not in y_val.columns:
    st.error("y_val.csv에는 returned 컬럼이 있어야 합니다.")
    st.stop()

if len(X_val) != len(y_val) or len(X_val) != len(val_pred):
    st.error("X_val_unscaled.csv, y_val.csv, val_predictions.csv의 행 개수가 서로 다릅니다.")
    st.stop()

val_df = X_val.copy()
val_df["returned"] = y_val["returned"].values
val_df["pred_prob"] = val_pred["pred_prob"].values
val_df["pred_label"] = val_pred["pred_label"].values
val_df = add_derived_columns(val_df)

# -----------------------------
# 제목
# -----------------------------
st.title("이커머스 반품 예측 및 운영 인사이트 대시보드")
st.caption("Validation 예측 결과와 변수 중요도를 기반으로 반품 패턴과 고위험 주문 특성을 분석합니다.")

# -----------------------------
# 사이드바 필터
# -----------------------------
st.sidebar.header("필터")

category_options = ["전체"] + sorted(val_df["product_category_label"].dropna().unique().tolist())
shipping_options = ["전체"] + sorted(val_df["shipping_method_label"].dropna().unique().tolist())
risk_options = ["전체", "Low", "Medium", "High"]

selected_category = st.sidebar.selectbox("카테고리", category_options)
selected_shipping = st.sidebar.selectbox("배송방식", shipping_options)
selected_risk = st.sidebar.selectbox("리스크 레벨", risk_options)

price_range = st.sidebar.slider(
    "상품 가격 범위",
    min_value=int(val_df["product_price"].min()),
    max_value=int(val_df["product_price"].max()),
    value=(int(val_df["product_price"].min()), int(val_df["product_price"].max()))
)

filtered_df = val_df.copy()

if selected_category != "전체":
    filtered_df = filtered_df[filtered_df["product_category_label"] == selected_category]

if selected_shipping != "전체":
    filtered_df = filtered_df[filtered_df["shipping_method_label"] == selected_shipping]

if selected_risk != "전체":
    filtered_df = filtered_df[filtered_df["risk_level"].astype(str) == selected_risk]

filtered_df = filtered_df[
    (filtered_df["product_price"] >= price_range[0]) &
    (filtered_df["product_price"] <= price_range[1])
]

# 재계산
f_accuracy, f_f1, f_precision, f_recall = calculate_metrics(filtered_df)

# -----------------------------
# KPI
# -----------------------------
st.subheader("핵심 지표")

k1, k2, k3, k4 = st.columns(4)
k1.metric("분석 대상 주문 수", f"{len(filtered_df):,}")
k2.metric("실제 반품률", f"{filtered_df['returned'].mean():.1%}" if len(filtered_df) > 0 else "0.0%")
k3.metric("평균 예측 확률", f"{filtered_df['pred_prob'].mean():.1%}" if len(filtered_df) > 0 else "0.0%")
k4.metric("고위험 주문 비율", f"{(filtered_df['pred_prob'] >= 0.7).mean():.1%}" if len(filtered_df) > 0 else "0.0%")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Accuracy", f"{f_accuracy:.4f}")
k6.metric("F1 Score", f"{f_f1:.4f}")
k7.metric("Precision", f"{f_precision:.4f}")
k8.metric("Recall", f"{f_recall:.4f}")

st.markdown("---")

# -----------------------------
# 예측 성능 및 분포
# -----------------------------
st.subheader("예측 결과 요약")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(filtered_df["pred_prob"], bins=20)
    ax.axvline(0.7, linestyle="--")
    apply_korean_font(ax, "예측 확률 분포", "예측 확률", "주문 수")
    st.pyplot(fig)

with row1_col2:
    risk_counts = filtered_df["risk_level"].value_counts().reindex(["Low", "Medium", "High"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(risk_counts.index.astype(str), risk_counts.values)
    apply_korean_font(ax, "리스크 레벨 분포", "리스크 레벨", "주문 수")
    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("실제값 vs 예측값 비교")

cm = pd.crosstab(
    filtered_df["returned"],
    filtered_df["pred_label"],
    rownames=["Actual"],
    colnames=["Predicted"]
)

c1, c2 = st.columns([1, 1.3])

with c1:
    st.dataframe(cm, use_container_width=True)

with c2:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm.values)
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns.tolist())
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index.tolist())
    apply_korean_font(ax, "혼동행렬", "예측값", "실제값")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center")

    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# 실제 반품 패턴 분석
# -----------------------------
st.subheader("실제 반품 패턴 분석")

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    category_rate = (
        filtered_df.groupby("product_category_label", as_index=False)["returned"]
        .mean()
        .sort_values("returned", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(category_rate["product_category_label"], category_rate["returned"])
    apply_korean_font(ax, "카테고리별 실제 반품률", "카테고리", "반품률")
    plt.xticks(rotation=20)
    st.pyplot(fig)

with row2_col2:
    shipping_rate = (
        filtered_df.groupby("shipping_method_label", as_index=False)["returned"]
        .mean()
        .sort_values("returned", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(shipping_rate["shipping_method_label"], shipping_rate["returned"])
    apply_korean_font(ax, "배송방식별 실제 반품률", "배송방식", "반품률")
    plt.xticks(rotation=20)
    st.pyplot(fig)

row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    discount_rate = (
        filtered_df.groupby("discount_bin", as_index=False, observed=False)["returned"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(discount_rate["discount_bin"].astype(str), discount_rate["returned"])
    apply_korean_font(ax, "할인율 구간별 실제 반품률", "할인율 구간", "반품률")
    st.pyplot(fig)

with row3_col2:
    delay_rate = (
        filtered_df.groupby("delay_days_int", as_index=False)["returned"]
        .mean()
        .sort_values("delay_days_int")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(delay_rate["delay_days_int"], delay_rate["returned"], marker="o")
    apply_korean_font(ax, "배송 지연일수별 실제 반품률", "배송 지연일수(정수)", "반품률")
    st.pyplot(fig)

row4_col1, row4_col2 = st.columns(2)

with row4_col1:
    past_return_rate = (
        filtered_df.groupby("past_return_bin", as_index=False, observed=False)["returned"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(past_return_rate["past_return_bin"].astype(str), past_return_rate["returned"])
    apply_korean_font(ax, "과거 반품률 구간별 실제 반품률", "과거 반품률 구간", "반품률")
    st.pyplot(fig)

with row4_col2:
    coupon_rate = (
        filtered_df.groupby("used_coupon", as_index=False)["returned"]
        .mean()
    )
    coupon_rate["coupon_label"] = coupon_rate["used_coupon"].map({0: "미사용", 1: "사용"})
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(coupon_rate["coupon_label"], coupon_rate["returned"])
    apply_korean_font(ax, "쿠폰 사용 여부별 실제 반품률", "쿠폰 사용 여부", "반품률")
    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# 리스크 기반 비교
# -----------------------------
st.subheader("리스크 그룹별 특성 비교")

r1, r2, r3 = st.columns(3)

with r1:
    risk_delay = (
        filtered_df.groupby("risk_level", observed=False)["delay_days_int"]
        .mean()
        .reindex(["Low", "Medium", "High"])
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(risk_delay.index.astype(str), risk_delay.values)
    apply_korean_font(ax, "리스크별 평균 배송지연", "리스크 레벨", "평균 배송지연일수")
    st.pyplot(fig)

with r2:
    risk_discount = (
        filtered_df.groupby("risk_level", observed=False)["discount_percent"]
        .mean()
        .reindex(["Low", "Medium", "High"])
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(risk_discount.index.astype(str), risk_discount.values)
    apply_korean_font(ax, "리스크별 평균 할인율", "리스크 레벨", "평균 할인율")
    st.pyplot(fig)

with r3:
    risk_past = (
        filtered_df.groupby("risk_level", observed=False)["past_return_rate"]
        .mean()
        .reindex(["Low", "Medium", "High"])
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(risk_past.index.astype(str), risk_past.values)
    apply_korean_font(ax, "리스크별 평균 과거 반품률", "리스크 레벨", "평균 과거 반품률")
    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("주요 영향 변수")

if {"feature", "importance"}.issubset(feature_importance.columns):
    plot_df = feature_importance.sort_values("importance", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df["importance"])
    apply_korean_font(ax, "Feature Importance TOP 10", "Importance", "")
    st.pyplot(fig)
else:
    st.warning("feature_importance.csv에는 feature, importance 컬럼이 있어야 합니다.")

st.markdown("---")

# -----------------------------
# 운영 인사이트
# -----------------------------
st.subheader("핵심 운영 인사이트")

insight_1 = filtered_df.loc[filtered_df["delay_days_int"] >= 2, "returned"].mean()
insight_2 = filtered_df.loc[filtered_df["discount_percent"] >= 40, "returned"].mean()
insight_3 = filtered_df.loc[filtered_df["past_return_rate"] >= 0.5, "returned"].mean()
insight_4 = filtered_df.loc[filtered_df["used_coupon"] == 1, "returned"].mean()

i1, i2 = st.columns(2)

with i1:
    st.markdown(f"""
- 배송 지연 **2일 이상** 주문의 실제 반품률: **{insight_1:.1%}**
- 할인율 **40% 이상** 주문의 실제 반품률: **{insight_2:.1%}**
""")

with i2:
    st.markdown(f"""
- 과거 반품률 **0.5 이상 고객**의 실제 반품률: **{insight_3:.1%}**
- 쿠폰 사용 주문의 실제 반품률: **{insight_4:.1%}**
""")

st.markdown("""
### 실행 전략 제안
- 배송 지연이 잦은 주문군은 물류 우선관리 대상으로 설정합니다.
- 과거 반품률이 높은 고객군은 구매 전 안내를 강화합니다.
- 고할인 상품은 상세페이지와 기대치 관리를 보완합니다.
- 반품률이 높은 카테고리는 리뷰, 옵션, 설명 정보 품질을 개선합니다.
""")

st.markdown("---")

# -----------------------------
# Test 예측 결과 요약
# -----------------------------
st.subheader("Test 예측 결과 요약")

if test_pred is not None and required_pred_cols.issubset(test_pred.columns):
    t1, t2, t3 = st.columns(3)
    t1.metric("Test 예측 건수", f"{len(test_pred):,}")
    t2.metric("평균 예측 확률", f"{test_pred['pred_prob'].mean():.1%}")
    t3.metric("고위험 예측 비율", f"{(test_pred['pred_prob'] >= 0.7).mean():.1%}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(test_pred["pred_prob"], bins=20)
    ax.axvline(0.7, linestyle="--")
    apply_korean_font(ax, "Test 예측 확률 분포", "예측 확률", "주문 수")
    st.pyplot(fig)
else:
    st.info("test_predictions.csv가 없거나 pred_prob, pred_label 컬럼이 없어 Test 요약은 생략했습니다.")

st.markdown("---")

# -----------------------------
# 상세 데이터 미리보기
# -----------------------------
st.subheader("Validation 데이터 미리보기")

preview_candidates = [
    "customer_age",
    "product_price",
    "discount_percent",
    "product_rating",
    "past_purchase_count",
    "past_return_rate",
    "delivery_delay_days",
    "delay_days_int",
    "is_delayed",
    "session_length_minutes",
    "num_product_views",
    "used_coupon",
    "product_category_label",
    "device_type_label",
    "shipping_method_label",
    "payment_method_label",
    "returned",
    "pred_prob",
    "pred_label",
    "risk_level",
]

preview_cols = [c for c in preview_candidates if c in filtered_df.columns]

st.dataframe(
    filtered_df[preview_cols].sort_values("pred_prob", ascending=False).head(30),
    use_container_width=True,
    hide_index=True
)

st.success("대시보드가 정상적으로 로드되었습니다.")