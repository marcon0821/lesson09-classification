import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. 페이지 설정
st.set_page_config(
    page_title="채점표 및 모델 평가 - 뇌졸중 예측 실습실", page_icon="📋", layout="wide"
)

# 데이터 URL
DATA_URL = "https://raw.githubusercontent.com/marcon0821/data/refs/heads/main/stroke.csv"
LOCAL_FILE = "stroke.csv"


# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL, encoding="utf-8")
        return df
    except Exception as e_url:
        try:
            df = pd.read_csv(LOCAL_FILE, encoding="utf-8")
            return df
        except Exception as e_local:
            st.error(
                f"❌ 데이터를 불러오지 못했습니다!\n\n"
                f"1. 웹 URL 오류: {e_url}\n"
                f"2. 로컬 파일 오류: {e_local}\n\n"
                f"GitHub 링크 주소를 확인하거나 `stroke.csv` 파일을 확인해주세요."
            )
            st.stop()


df = load_data()

# 2. 페이지 제목
st.title("📋 뇌졸중 예측 모델 채점표 및 평가")
st.markdown(
    "테스트 데이터를 바탕으로 두 모델(로지스틱 회귀, 의사결정트리)의 혼동행렬과 성능 지표를 상세히 채점하고 비교합니다."
)
st.divider()

# 속성 매핑 사전
feature_mapping = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
reverse_feature_mapping = {v: k for k, v in feature_mapping.items()}
all_korean_features = list(feature_mapping.values())
default_korean_features = [
    feature_mapping[f] for f in ["age", "avg_glucose_level", "hypertension", "heart_disease"]
]

# 모델 입력 속성 선택 UI
st.subheader("⚙️ 모델 입력 속성 선택")
selected_korean_features = st.multiselect(
    "학습 및 예측에 사용할 속성을 선택해 주세요 (2개 이상)",
    options=all_korean_features,
    default=default_korean_features,
)

if len(selected_korean_features) < 2:
    st.warning("⚠️ 모델을 평가하려면 **속성을 2개 이상** 선택해 주세요.")
    st.stop()

selected_features = [reverse_feature_mapping[k] for k in selected_korean_features]

# 데이터 전처리 및 분할 규칙 적용 (분류 모델 페이지와 동일)
df_sorted = df.sort_values(by="id").reset_index(drop=True)

test_indices = []
train_indices = []

for i in range(0, len(df_sorted), 10):
    chunk = df_sorted.iloc[i : i + 10]
    if len(chunk) >= 3:
        test_indices.extend(chunk.index[:3])
        train_indices.extend(chunk.index[3:])
    else:
        test_indices.extend(chunk.index[:])

df_train = df_sorted.loc[train_indices].copy()
df_test = df_sorted.loc[test_indices].copy()

# bmi 결측치 처리 (훈련용 중앙값으로 채우기)
if "bmi" in selected_features:
    bmi_median = df_train["bmi"].median()
    df_train["bmi"] = df_train["bmi"].fillna(bmi_median)
    df_test["bmi"] = df_test["bmi"].fillna(bmi_median)

X_train = df_train[selected_features]
y_train = df_train["stroke"]
X_test = df_test[selected_features]
y_test = df_test["stroke"]

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
# 1) 로지스틱 회귀 (확률로 답하는 모델)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
log_preds = log_reg.predict(X_test_scaled)

# 2) 의사결정트리 (질문으로 답하는 모델)
dt_clf = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
dt_clf.fit(X_train, y_train)
dt_preds = dt_clf.predict(X_test)

# 3) 베이스라인 모델 (훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = y_train.mode()[0]
dummy_preds = np.full_like(y_test, majority_class)

st.divider()

# 테스트 데이터 정보 표시
total_test_count = len(y_test)
actual_stroke_test_count = int(y_test.sum())
st.markdown(
    f"📌 **테스트용 사람 수:** 총 **{total_test_count:,}명** | **그중 실제 뇌졸중인 사람 수:** **{actual_stroke_test_count:,}명**"
)

st.divider()

# 혼동행렬 (Confusion Matrix)
st.subheader("📊 1. 혼동행렬 (실제 vs 예측)")
st.markdown(
    "행은 **실제 값** (0: 정상, 1: 뇌졸중), 열은 **예측 값** (0: 정상, 1: 뇌졸중)을 나타내며 각 칸에 TP, FP, FN, TN 표기를 포함합니다."
)


def get_cm_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


log_tn, log_fp, log_fn, log_tp = get_cm_metrics(y_test, log_preds)
dt_tn, dt_fp, dt_fn, dt_tp = get_cm_metrics(y_test, dt_preds)
dummy_tn, dummy_fp, dummy_fn, dummy_tp = get_cm_metrics(y_test, dummy_preds)

cm_col1, cm_col2 = st.columns(2)

with cm_col1:
    st.markdown("#####로지스틱 회귀(확률로 답하는 모델)")
    cm_log_df = pd.DataFrame(
        {
            "예측: 정상(0)": [f"TN: {log_tn:,}", f"FN: {log_fn:,}", f"합계: {log_tn + log_fn:,}"],
            "예측: 뇌졸중(1)": [f"FP: {log_fp:,}", f"TP: {log_tp:,}", f"합계: {log_fp + log_tp:,}"],
            "합계": [f"{log_tn + log_fp:,}", f"{log_fn + log_tp:,}", f"{total_test_count:,}"],
        },
        index=["실제: 정상(0)", "실제: 뇌졸중(1)", "합계"],
    )
    st.dataframe(cm_log_df, use_container_width=True)

with cm_col2:
    st.markdown("##### 의사결정트리(질문으로 답하는 모델)")
    cm_dt_df = pd.DataFrame(
        {
            "예측: 정상(0)": [f"TN: {dt_tn:,}", f"FN: {dt_fn:,}", f"합계: {dt_tn + dt_fn:,}"],
            "예측: 뇌졸중(1)": [f"FP: {dt_fp:,}", f"TP: {dt_tp:,}", f"합계: {dt_fp + dt_tp:,}"],
            "합계": [f"{dt_tn + dt_fp:,}", f"{dt_fn + dt_tp:,}", f"{total_test_count:,}"],
        },
        index=["실제: 정상(0)", "실제: 뇌졸중(1)", "합계"],
    )
    st.dataframe(cm_dt_df, use_container_width=True)

st.divider()

# 성능 지표 표 (정확도, 재현율, 정밀도, F1)
st.subheader("📋 2. 주요 성능 지표 비교 표")
st.markdown(
    "재현율은 실제 뇌졸중인 사람, 정밀도는 뇌졸중이라 예측한 사람을 분모로 계산하며, 예측 환자가 0명인 경우 '계산할 수 없음'으로 표시합니다."
)


def calculate_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    actual_positive = tp + fn
    recall = tp / actual_positive if actual_positive > 0 else 0

    predicted_positive = tp + fp
    if predicted_positive > 0:
        precision = tp / predicted_positive
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        precision_str = f"{precision * 100:.2f}%"
        f1_str = f"{f1:.4f}"
    else:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"

    return (
        f"{accuracy * 100:.2f}%",
        f"{recall * 100:.2f}%",
        precision_str,
        f1_str,
    )


log_acc, log_rec, log_prec, log_f1 = calculate_metrics(log_tn, log_fp, log_fn, log_tp)
dt_acc, dt_rec, dt_prec, dt_f1 = calculate_metrics(dt_tn, dt_fp, dt_fn, dt_tp)
dummy_acc, dummy_rec, dummy_prec, dummy_f1 = calculate_metrics(
    dummy_tn, dummy_fp, dummy_fn, dummy_tp
)

metrics_df = pd.DataFrame(
    {
        "모델 이름": [
            "로지스틱 회귀(확률로 답하는 모델)",
            "의사결정트리(질문으로 답하는 모델)",
            "훈련용에서 많은 쪽으로만 답하는 모델",
        ],
        "정확도 (Accuracy)": [log_acc, dt_acc, dummy_acc],
        "재현율 (Recall)": [log_rec, dt_rec, dummy_rec],
        "정밀도 (Precision)": [log_prec, dt_prec, dummy_prec],
        "F1-스코어": [log_f1, dt_f1, dummy_f1],
    }
)

st.dataframe(metrics_df, use_container_width=True)

st.divider()

# 오류 분석 목록 (놓친 환자 및 헛짚은 사람)
st.subheader("🔍 3. 오류 분석 목록 (놓친 환자 vs 헛짚은 사람)")
st.markdown(
    "어떤 모델의 오류 목록을 확인할지 선택하세요. 각 목록은 **나이가 많은 순**으로 정렬되며, 20줄을 넘는 경우 앞 20줄만 보여줍니다."
)

model_choice = st.selectbox(
    "분석할 모델 선택",
    options=[
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
    ],
)

chosen_preds = log_preds if "로지스틱" in model_choice else dt_preds

df_error_analysis = df_test.copy()
df_error_analysis["predicted"] = chosen_preds

# 왼쪽: 실제 뇌졸중인데 아니라고 예측 (FN - 놓친 환자)
df_fn_list = df_error_analysis[
    (df_error_analysis["stroke"] == 1) & (df_error_analysis["predicted"] == 0)
].sort_values(by="age", ascending=False)

# 오른쪽: 실제 정상인데 뇌졸중이라고 예측 (FP - 헛짚은 사람)
df_fp_list = df_error_analysis[
    (df_error_analysis["stroke"] == 0) & (df_error_analysis["predicted"] == 1)
].sort_values(by="age", ascending=False)

fn_total_count = len(df_fn_list)
fp_total_count = len(df_fp_list)

col_err1, col_err2 = st.columns(2)

columns_to_show = ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]
column_renames = {
    "id": "번호",
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부",
}

with col_err1:
    st.markdown(f"##### 🚨 놓친 환자 (총 {fn_total_count}명)")
    if fn_total_count > 20:
        st.caption("⚠️ 20줄을 초과하여 앞 20줄만 표시합니다.")
        display_fn = df_fn_list.head(20)
    else:
        display_fn = df_fn_list

    display_fn_table = display_fn[columns_to_show].rename(columns=column_renames)
    display_fn_table["고혈압 여부"] = display_fn_table["고혈압 여부"].map(
        {0: "없음(0)", 1: "있음(1)"}
    )
    display_fn_table["심장병 여부"] = display_fn_table["심장병 여부"].map(
        {0: "없음(0)", 1: "있음(1)"}
    )
    st.dataframe(display_fn_table.reset_index(drop=True), use_container_width=True)

with col_err2:
    st.markdown(f"##### ⚠️ 헛짚은 사람 (총 {fp_total_count}명)")
    if fp_total_count > 20:
        st.caption("⚠️ 20줄을 초과하여 앞 20줄만 표시합니다.")
        display_fp = df_fp_list.head(20)
    else:
        display_fp = df_fp_list

    display_fp_table = display_fp[columns_to_show].rename(columns=column_renames)
    display_fp_table["고혈압 여부"] = display_fp_table["고혈압 여부"].map(
        {0: "없음(0)", 1: "있음(1)"}
    )
    display_fp_table["심장병 여부"] = display_fp_table["심장병 여부"].map(
        {0: "없음(0)", 1: "있음(1)"}
    )
    st.dataframe(display_fp_table.reset_index(drop=True), use_container_width=True)

st.divider()

# 혼동행렬의 FN, FP와 인원수 일치 안내
chosen_fn = log_fn if "로지스틱" in model_choice else dt_fn
chosen_fp = log_fp if "로지스틱" in model_choice else dt_fp
st.info(
    f"💡 **확인:** 위 두 목록의 인원 수(놓친 환자 **{fn_total_count}명**, 헛짚은 사람 **{fp_total_count}명**)는 "
    f"선택하신 모델의 혼동행렬 내 **FN 칸({chosen_fn}명)** 및 **FP 칸({chosen_fp}명)**과 정확히 일치합니다."
)
