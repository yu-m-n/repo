import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'

st.set_page_config(page_title="반품 예측 대시보드 예시", layout="wide")

# =============================
# 1. 예시 데이터 생성
# =============================
@st.cache_data
def make_mock_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "product_category": rng.choice(["전자", "의류", "식품", "뷰티", "생활"], size=n),
        "product_price": rng.integers(10000, 250000, size=n),
        "discount_percent": rng.integers(0, 71, size=n),
        "delivery_delay_days": rng.integers(0, 8, size=n),
        "past_return_rate": np.round(rng.uniform(0, 1, size=n), 2),
        "used_coupon": rng.choice([0, 1], size=n, p=[0.4, 0.6]),
        "customer_tier": rng.choice(["Bronze", "Silver", "Gold"], size=n, p=[0.4, 0.4, 0.2]),
    })

    # 예측 확률을 어느 정도 그럴듯하게 만들기 위한 간단한 규칙
    base = (
        0.15
        + 0.35 * df["past_return_rate"]
        + 0.06 * (df["delivery_delay_days"] >= 3).astype(float)
        + 0.05 * (df["discount_percent"] >= 40).astype(float)
        + 0.04 * (df["used_coupon"] == 1).astype(float)
    )

    category_effect = df["product_category"].map({
        "전자": 0.06,
        "의류": 0.08,
        "식품": -0.03,
        "뷰티": 0.02,
        "생활": 0.00,
    }).astype(float)

    noise = rng.normal(0, 0.06, size=n)

    df["pred_prob"] = np.clip(base + category_effect + noise, 0.01, 0.99)
    threshold = 0.50
    df["pred_label"] = (df["pred_prob"] >= threshold).astype(int)

    df["risk_level"] = pd.cut(
        df["pred_prob"],
        bins=[0.0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    # 실제값도 예시용으로 생성
    df["actual_returned"] = rng.binomial(1, df["pred_prob"])
    return df


@st.cache_data
def make_feature_importance():
    return pd.DataFrame({
        "feature": [
            "past_return_rate",
            "delivery_delay_days",
            "discount_percent",
            "product_category_의류",
            "product_category_전자",
            "used_coupon",
            "product_price",
        ],
        "importance": [0.31, 0.22, 0.15, 0.11, 0.09, 0.06, 0.04],
    }).sort_values("importance", ascending=False)


# =============================
# 2. 데이터 준비
# =============================
df = make_mock_data()
feature_importance = make_feature_importance()

st.title("반품 예측 대시보드 예시")
st.caption("팀원의 실제 모델 결과가 오기 전, 구조를 미리 연습하기 위한 예시 화면입니다.")

# =============================
# 3. 사이드바 필터
# =============================
st.sidebar.header("필터")
category_options = ["전체"] + sorted(df["product_category"].unique().tolist())
selected_category = st.sidebar.selectbox("상품 카테고리", category_options)
selected_risk = st.sidebar.multiselect(
    "리스크 레벨",
    options=["Low", "Medium", "High"],
    default=["Low", "Medium", "High"],
)
price_range = st.sidebar.slider(
    "상품 가격 범위",
    min_value=int(df["product_price"].min()),
    max_value=int(df["product_price"].max()),
    value=(int(df["product_price"].min()), int(df["product_price"].max())),
)

filtered = df.copy()
if selected_category != "전체":
    filtered = filtered[filtered["product_category"] == selected_category]
filtered = filtered[filtered["risk_level"].astype(str).isin(selected_risk)]
filtered = filtered[
    (filtered["product_price"] >= price_range[0])
    & (filtered["product_price"] <= price_range[1])
]

# =============================
# 4. KPI 영역
# =============================
total_orders = len(filtered)
pred_return_rate = filtered["pred_label"].mean() if total_orders else 0
high_risk_count = (filtered["risk_level"].astype(str) == "High").sum()
avg_pred_prob = filtered["pred_prob"].mean() if total_orders else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 주문 수", f"{total_orders:,}")
col2.metric("예상 반품률", f"{pred_return_rate:.1%}")
col3.metric("고위험 주문 수", f"{high_risk_count:,}")
col4.metric("평균 예측 확률", f"{avg_pred_prob:.1%}")

# =============================
# 5. 차트 영역
# =============================
left, right = st.columns(2)

with left:
    st.subheader("리스크 레벨 분포")
    risk_counts = filtered["risk_level"].value_counts().reindex(["Low", "Medium", "High"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(risk_counts.index.astype(str), risk_counts.values)
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("주문 수")
    st.pyplot(fig)

with right:
    st.subheader("예측 확률 분포")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(filtered["pred_prob"], bins=20)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    st.pyplot(fig)

left2, right2 = st.columns(2)

with left2:
    st.subheader("카테고리별 예상 반품률")
    category_rate = (
        filtered.groupby("product_category", as_index=False)["pred_label"]
        .mean()
        .sort_values("pred_label", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(category_rate["product_category"], category_rate["pred_label"])
    ax.set_xlabel("Category")
    ax.set_ylabel("Predicted Return Rate")
    plt.xticks(rotation=30)
    st.pyplot(fig)

with right2:
    st.subheader("배송 지연일수별 예상 반품률")
    delay_rate = (
        filtered.groupby("delivery_delay_days", as_index=False)["pred_label"]
        .mean()
        .sort_values("delivery_delay_days")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(delay_rate["delivery_delay_days"], delay_rate["pred_label"], marker="o")
    ax.set_xlabel("Delivery Delay Days")
    ax.set_ylabel("Predicted Return Rate")
    st.pyplot(fig)

# =============================
# 6. 중요 변수 영역
# =============================
st.subheader("주요 영향 요인 (예시 Feature Importance)")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(feature_importance["feature"], feature_importance["importance"])
ax.set_xlabel("Feature")
ax.set_ylabel("Importance")
plt.xticks(rotation=30, ha="right")
st.pyplot(fig)

# =============================
# 7. 액션 인사이트 영역
# =============================
st.subheader("운영 인사이트 예시")
insight_1 = filtered[filtered["delivery_delay_days"] >= 3]["pred_label"].mean() if len(filtered[filtered["delivery_delay_days"] >= 3]) else 0
insight_2 = filtered[filtered["discount_percent"] >= 40]["pred_label"].mean() if len(filtered[filtered["discount_percent"] >= 40]) else 0
insight_3 = filtered[filtered["past_return_rate"] >= 0.5]["pred_label"].mean() if len(filtered[filtered["past_return_rate"] >= 0.5]) else 0

st.markdown(f"- 배송 지연이 **3일 이상**인 주문의 예상 반품률: **{insight_1:.1%}**")
st.markdown(f"- 할인율이 **40% 이상**인 주문의 예상 반품률: **{insight_2:.1%}**")
st.markdown(f"- 과거 반품률이 **0.5 이상**인 고객의 예상 반품률: **{insight_3:.1%}**")

# =============================
# 8. 상세 주문 테이블
# =============================
st.subheader("상세 주문 목록")
show_cols = [
    "order_id",
    "product_category",
    "product_price",
    "discount_percent",
    "delivery_delay_days",
    "past_return_rate",
    "pred_prob",
    "pred_label",
    "risk_level",
]
st.dataframe(
    filtered[show_cols].sort_values("pred_prob", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.info(
    "나중에 팀원이 준 실제 결과 파일에서 pred_prob, pred_label, feature importance만 바꿔 끼우면 이 구조를 그대로 재사용할 수 있습니다."
)
