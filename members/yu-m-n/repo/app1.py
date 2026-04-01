from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[2]

st.set_page_config(page_title="반품 예측 대시보드", layout="wide")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


@st.cache_data
def load_data():
    X_tr_unscaled = pd.read_csv(ROOT_DIR / "X_tr_unscaled.csv")
    X_val_unscaled = pd.read_csv(ROOT_DIR / "X_val_unscaled.csv")
    X_test_unscaled = pd.read_csv(ROOT_DIR / "X_test_unscaled.csv")
    y_tr = pd.read_csv(ROOT_DIR / "y_tr.csv")
    y_val = pd.read_csv(ROOT_DIR / "y_val.csv")
    test_order_id = pd.read_csv(ROOT_DIR / "test_order_id.csv")

    metrics = pd.DataFrame([
        {"model": "LightGBM", "accuracy": 0.5673, "f1": 0.4915, "roc_auc": 0.5927, "note": "ROC-AUC 최고"},
        {"model": "CatBoost", "accuracy": 0.5680, "f1": 0.4909, "roc_auc": 0.5926, "note": "Accuracy 최고"},
        {"model": "XGBoost", "accuracy": 0.5658, "f1": 0.4999, "roc_auc": 0.5879, "note": "F1 최고"},
        {"model": "Logistic Regression", "accuracy": 0.5656, "f1": 0.4853, "roc_auc": 0.5876, "note": "선형 기준 모델"},
    ])

    return X_tr_unscaled, X_val_unscaled, X_test_unscaled, y_tr, y_val, test_order_id, metrics


def decode_onehot(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    if not cols:
        return pd.Series(["unknown"] * len(df), index=df.index)
    return df[cols].idxmax(axis=1).str.replace(prefix + "_", "", regex=False)


X_tr, X_val, X_test, y_tr, y_val, test_order_id, metrics_df = load_data()

best_auc = metrics_df.loc[metrics_df["roc_auc"].idxmax()]
best_f1 = metrics_df.loc[metrics_df["f1"].idxmax()]
best_acc = metrics_df.loc[metrics_df["accuracy"].idxmax()]

st.title("반품 예측 프로젝트 대시보드")
st.caption("실제 전처리 데이터셋과 실제 모델 성능 결과를 기반으로 구성한 대시보드")

# -------------------------------------------------
# 사이드바
# -------------------------------------------------
st.sidebar.header("대시보드 설정")
metric_choice = st.sidebar.selectbox(
    "모델 비교 기준",
    ["roc_auc", "f1", "accuracy"],
    format_func=lambda x: {
        "roc_auc": "ROC-AUC",
        "f1": "F1 Score",
        "accuracy": "Accuracy"
    }[x],
)

# -------------------------------------------------
# 상단 KPI
# -------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Train 데이터 수", f"{len(X_tr):,}")
k2.metric("Validation 데이터 수", f"{len(X_val):,}")
k3.metric("Test 데이터 수", f"{len(X_test):,}")
k4.metric("Validation 실제 반품률", f"{y_val['returned'].mean():.1%}")

st.markdown("---")

# -------------------------------------------------
# 모델 성능 비교
# -------------------------------------------------
st.subheader("모델 성능 비교")

left, right = st.columns([1.2, 1])

with left:
    st.dataframe(
        metrics_df.sort_values(metric_choice, ascending=False),
        width="stretch",
        hide_index=True,
    )

with right:
    sorted_df = metrics_df.sort_values(metric_choice, ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sorted_df["model"], sorted_df[metric_choice])
    ax.set_xlabel("모델")
    ax.set_ylabel(metric_choice.upper())
    plt.xticks(rotation=20)
    st.pyplot(fig)

c1, c2, c3 = st.columns(3)
c1.metric("최고 ROC-AUC", best_auc["model"], f"{best_auc['roc_auc']:.4f}")
c2.metric("최고 F1", best_f1["model"], f"{best_f1['f1']:.4f}")
c3.metric("최고 Accuracy", best_acc["model"], f"{best_acc['accuracy']:.4f}")

st.info(
    "운영형 관점에서는 ROC-AUC가 가장 높은 LightGBM을 대표 모델로 설명하고, "
    "분류 균형 관점에서는 F1이 가장 높은 XGBoost를 함께 비교하는 방식이 자연스럽습니다."
)

st.markdown("---")

# -------------------------------------------------
# Train / Validation 타깃 분포
# -------------------------------------------------
st.subheader("반품 타깃 분포")

dist1, dist2 = st.columns(2)

with dist1:
    dist_df = pd.DataFrame({
        "dataset": ["Train", "Validation"],
        "return_rate": [y_tr["returned"].mean(), y_val["returned"].mean()]
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(dist_df["dataset"], dist_df["return_rate"])
    ax.set_ylabel("반품률")
    ax.set_title("Train / Validation 반품률")
    st.pyplot(fig)

with dist2:
    val_counts = y_val["returned"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["0", "1"], val_counts.values)
    ax.set_xlabel("returned")
    ax.set_ylabel("건수")
    ax.set_title("Validation 타깃 분포")
    st.pyplot(fig)

st.markdown("---")

# -------------------------------------------------
# Validation 실제 데이터 기반 분석
# -------------------------------------------------
st.subheader("Validation 실제 데이터 분석")

val_df = X_val.copy()
val_df["returned"] = y_val["returned"].values

# 범주형 복원
val_df["product_category_label"] = decode_onehot(val_df, "product_category")
val_df["device_type_label"] = decode_onehot(val_df, "device_type")
val_df["shipping_method_label"] = decode_onehot(val_df, "shipping_method")
val_df["payment_method_label"] = decode_onehot(val_df, "payment_method")

category_options = ["전체"] + sorted(val_df["product_category_label"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("카테고리 필터", category_options)

filtered_df = val_df.copy()
if selected_category != "전체":
    filtered_df = filtered_df[filtered_df["product_category_label"] == selected_category]

v1, v2, v3, v4 = st.columns(4)
v1.metric("분석 대상 주문 수", f"{len(filtered_df):,}")
v2.metric("실제 반품률", f"{filtered_df['returned'].mean():.1%}")
v3.metric("평균 할인율", f"{filtered_df['discount_percent'].mean():.1f}%")
v4.metric("평균 배송지연일", f"{filtered_df['delivery_delay_days'].mean():.2f}")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    category_rate = (
        filtered_df.groupby("product_category_label", as_index=False)["returned"]
        .mean()
        .sort_values("returned", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(category_rate["product_category_label"], category_rate["returned"])
    ax.set_title("카테고리별 실제 반품률")
    ax.set_xlabel("카테고리")
    ax.set_ylabel("반품률")
    plt.xticks(rotation=20)
    st.pyplot(fig)

with row1_col2:
    shipping_rate = (
        filtered_df.groupby("shipping_method_label", as_index=False)["returned"]
        .mean()
        .sort_values("returned", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(shipping_rate["shipping_method_label"], shipping_rate["returned"])
    ax.set_title("배송방식별 실제 반품률")
    ax.set_xlabel("배송방식")
    ax.set_ylabel("반품률")
    plt.xticks(rotation=20)
    st.pyplot(fig)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    tmp = filtered_df.copy()
    tmp["discount_bin"] = pd.cut(
        tmp["discount_percent"],
        bins=[-0.1, 10, 30, 50, 100],
        labels=["0-10", "10-30", "30-50", "50+"]
    )
    discount_rate = tmp.groupby("discount_bin", as_index=False)["returned"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(discount_rate["discount_bin"].astype(str), discount_rate["returned"])
    ax.set_title("할인율 구간별 실제 반품률")
    ax.set_xlabel("할인율 구간")
    ax.set_ylabel("반품률")
    st.pyplot(fig)

with row2_col2:
    delay_rate = (
        filtered_df.groupby("delivery_delay_days", as_index=False)["returned"]
        .mean()
        .sort_values("delivery_delay_days")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(delay_rate["delivery_delay_days"], delay_rate["returned"], marker="o")
    ax.set_title("배송 지연일수별 실제 반품률")
    ax.set_xlabel("배송 지연일수")
    ax.set_ylabel("반품률")
    st.pyplot(fig)

st.markdown("---")

# -------------------------------------------------
# 운영 인사이트
# -------------------------------------------------
st.subheader("운영 인사이트")

insight_1 = filtered_df.loc[filtered_df["delivery_delay_days"] >= 2, "returned"].mean()
insight_2 = filtered_df.loc[filtered_df["discount_percent"] >= 40, "returned"].mean()
insight_3 = filtered_df.loc[filtered_df["past_return_rate"] >= 0.5, "returned"].mean()
insight_4 = filtered_df.loc[filtered_df["used_coupon"] == 1, "returned"].mean()

i1, i2 = st.columns(2)

with i1:
    st.markdown(f"""
- 배송 지연 **2일 이상** 주문 실제 반품률: **{insight_1:.1%}**
- 할인율 **40% 이상** 주문 실제 반품률: **{insight_2:.1%}**
""")

with i2:
    st.markdown(f"""
- 과거 반품률 **0.5 이상 고객** 실제 반품률: **{insight_3:.1%}**
- 쿠폰 사용 주문 실제 반품률: **{insight_4:.1%}**
""")

st.warning(
    "현재 업로드된 파일에는 test 예측 확률(pred_prob) 결과 파일이 없어서, "
    "고위험 주문 우선관리 리스트는 아직 만들 수 없습니다. "
    "팀원이 order_id + pred_prob 파일을 주면 마지막 섹션에 바로 추가할 수 있습니다."
)

st.markdown("---")

# -------------------------------------------------
# 데이터 미리보기
# -------------------------------------------------
st.subheader("Validation 데이터 미리보기")

preview_cols = [
    c for c in [
        "product_price",
        "discount_percent",
        "product_rating",
        "past_return_rate",
        "session_length_minutes",
        "num_product_views",
        "delivery_delay_days",
        "used_coupon",
        "product_category_label",
        "device_type_label",
        "shipping_method_label",
        "payment_method_label",
        "returned",
    ] if c in filtered_df.columns
]

st.dataframe(filtered_df[preview_cols].head(30), width="stretch", hide_index=True)

st.success("이 대시보드는 학습 없이, 실제 업로드된 데이터와 이미 계산된 성능 결과만 사용합니다.")