import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. 페이지 설정
st.set_page_config(
    page_title="모델 성능 높이기 (가중치 조정) - 뇌졸중 예측 실습실", page_icon="⚖️", layout="wide"
)

# 데이터 URL (최신 고정 링크)
DATA_URL = "https://raw.githubusercontent.com/marcon0821/data/refs/main/stroke.csv"
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
st.title("⚖️ 모델 성능 높이기: 클래스 가중치 조정")
st.markdown(
    "뇌졸중 환자 데이터가 아주 드문 불균형 데이터셋일 때, 클래스 가중치를 맞추기 전과 맞춘 후의 모델 성능과 결정 경계를 비교합니다."
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

# 3. 입력 속성 선택 UI (분류 모델 페이지와 동일)
st.subheader("⚙️ 모델 입력 속성 선택")
selected_korean_features = st.multiselect(
    "학습 및 예측에 사용할 속성을 선택해 주세요 (2개 이상)",
    options=all_korean_features,
    default=default_korean_features,
)

if len(selected_korean_features) < 2:
    st.warning("⚠️ 모델을 학습하려면 **속성을 2개 이상** 선택해 주세요.")
    st.stop()

selected_features = [reverse_feature_mapping[k] for k in selected_korean_features]

# 데이터 전처리 및 분할 규칙 적용
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


# 4. 모델 학습 (가중치 전 / 후)
def get_metrics_dict(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total > 0 else 0
    actual_pos = tp + fn
    recall = tp / actual_pos if actual_pos > 0 else 0
    pred_pos = tp + fp
    if pred_pos > 0:
        prec = tp / pred_pos
        f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
        prec_str = f"{prec * 100:.2f}%"
        f1_str = f"{f1:.4f}"
    else:
        prec_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "acc": f"{acc * 100:.2f}%",
        "recall": f"{recall * 100:.2f}%",
        "precision": prec_str,
        "f1": f1_str,
    }


# 1) 로지스틱 회귀 (가중치 전 vs 후)
log_reg_before = LogisticRegression(random_state=42)
log_reg_before.fit(X_train_scaled, y_train)
log_pred_b = log_reg_before.predict(X_test_scaled)
log_m_b = get_metrics_dict(y_test, log_pred_b)

log_reg_after = LogisticRegression(class_weight="balanced", random_state=42)
log_reg_after.fit(X_train_scaled, y_train)
log_pred_a = log_reg_after.predict(X_test_scaled)
log_m_a = get_metrics_dict(y_test, log_pred_a)

# 2) 의사결정트리 (가중치 전 vs 후)
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train, y_train)
dt_pred_b = dt_before.predict(X_test)
dt_m_b = get_metrics_dict(y_test, dt_pred_b)

dt_after = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42
)
dt_after.fit(X_train, y_train)
dt_pred_a = dt_after.predict(X_test)
dt_m_a = get_metrics_dict(y_test, dt_pred_a)

st.divider()

# 5. 혼동행렬 및 성능 지표 비교 표 (전·후 나란히)
st.subheader("📊 모델 가중치 조정 전·후 성능 비교 표")
st.markdown("가중치 조절 전(`None`)과 조절 후(`balanced`)의 혼동행렬 지표와 4가지 성능 지표를 나란히 비교합니다.")

col_tab1, col_tab2 = st.columns(2)

with col_tab1:
    st.markdown("##### 📈 로지스틱 회귀(확률로 답하는 모델)")
    log_comp_df = pd.DataFrame(
        {
            "조정 전 (기본)": [
                f"TN: {log_m_b['tn']}",
                f"FP: {log_m_b['fp']}",
                f"FN: {log_m_b['fn']}",
                f"TP: {log_m_b['tp']}",
                log_m_b["acc"],
                log_m_b["recall"],
                log_m_b["precision"],
                log_m_b["f1"],
            ],
            "조정 후 (가중치 균형)": [
                f"TN: {log_m_a['tn']}",
                f"FP: {log_m_a['fp']}",
                f"FN: {log_m_a['fn']}",
                f"TP: {log_m_a['tp']}",
                log_m_a["acc"],
                log_m_a["recall"],
                log_m_a["precision"],
                log_m_a["f1"],
            ],
        },
        index=[
            "참 음성 (TN)",
            "거짓 양성 (FP)",
            "거짓 음성 (FN)",
            "참 양성 (TP)",
            "정확도 (Accuracy)",
            "재현율 (Recall)",
            "정밀도 (Precision)",
            "F1-스코어",
        ],
    )
    st.dataframe(log_comp_df, use_container_width=True)
    st.markdown(
        f"📝 **변동 요약:** 찾아낸 환자 수(TP) **{log_m_b['tp']}명 ➔ {log_m_a['tp']}명**, "
        f"놓친 환자 수(FN) **{log_m_b['fn']}명 ➔ {log_m_a['fn']}명**, "
        f"헛짚은 사람 수(FP) **{log_m_b['fp']}명 ➔ {log_m_a['fp']}명**으로 변경되었습니다."
    )

with col_tab2:
    st.markdown("##### 🌳 의사결정트리(질문으로 답하는 모델)")
    dt_comp_df = pd.DataFrame(
        {
            "조정 전 (기본)": [
                f"TN: {dt_m_b['tn']}",
                f"FP: {dt_m_b['fp']}",
                f"FN: {dt_m_b['fn']}",
                f"TP: {dt_m_b['tp']}",
                dt_m_b["acc"],
                dt_m_b["recall"],
                dt_m_b["precision"],
                dt_m_b["f1"],
            ],
            "조정 후 (가중치 균형)": [
                f"TN: {dt_m_a['tn']}",
                f"FP: {dt_m_a['fp']}",
                f"FN: {dt_m_a['fn']}",
                f"TP: {dt_m_a['tp']}",
                dt_m_a["acc"],
                dt_m_a["recall"],
                dt_m_a["precision"],
                dt_m_a["f1"],
            ],
        },
        index=[
            "참 음성 (TN)",
            "거짓 양성 (FP)",
            "거짓 음성 (FN)",
            "참 양성 (TP)",
            "정확도 (Accuracy)",
            "재현율 (Recall)",
            "정밀도 (Precision)",
            "F1-스코어",
        ],
    )
    st.dataframe(dt_comp_df, use_container_width=True)
    st.markdown(
        f"📝 **변동 요약:** 찾아낸 환자 수(TP) **{dt_m_b['tp']}명 ➔ {dt_m_a['tp']}명**, "
        f"놓친 환자 수(FN) **{dt_m_b['fn']}명 ➔ {dt_m_a['fn']}명**, "
        f"헛짚은 사람 수(FP) **{dt_m_b['fp']}명 ➔ {dt_m_a['fp']}명**으로 변경되었습니다."
    )

st.divider()

# 6. 결정 경계 시각화 (2차원 / 3차원)
st.subheader("📈 결정 경계선 비교 시각화 (로지스틱 회귀 기준)")

if len(selected_korean_features) == 2:
    st.markdown("속성을 2개 선택하셨으므로 2차원 산점도에 가중치 조정 전후의 0.5 경계선을 겹쳐서 표시합니다.")

    c1, c2 = st.columns(2)
    with c1:
        plot_x_korean = st.selectbox("가로축 속성", options=selected_korean_features, index=0)
    with c2:
        plot_y_korean = st.selectbox("세로축 속성", options=selected_korean_features, index=1)

    plot_x_feat = reverse_feature_mapping[plot_x_korean]
    plot_y_feat = reverse_feature_mapping[plot_y_korean]

    if plot_x_feat == plot_y_feat:
        st.warning("⚠️ 가로축과 세로축에 서로 다른 속성을 선택해 주세요.")
    else:
        other_features = [f for f in selected_features if f not in [plot_x_feat, plot_y_feat]]
        fixed_values = {f: df_train[f].median() for f in other_features}

        df_test_plot = df_test.copy()
        df_test_plot["실제_뇌졸중"] = df_test_plot["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

        fig = px.scatter(
            df_test_plot,
            x=plot_x_feat,
            y=plot_y_feat,
            color="실제_뇌졸중",
            labels={plot_x_feat: plot_x_korean, plot_y_feat: plot_y_korean},
            title="가중치 조정 전 vs 후 로지스틱 회귀 0.5 경계선 비교",
            color_discrete_map={"정상 (0)": "#636EFA", "뇌졸중 (1)": "#EF553B"},
        )

        x_min, x_max = df_test[plot_x_feat].min(), df_test[plot_x_feat].max()
        y_min, y_max = df_test[plot_y_feat].min(), df_test[plot_y_feat].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        grid_data = pd.DataFrame(index=np.arange(xx.ravel().shape[0]))
        for f in selected_features:
            if f == plot_x_feat:
                grid_data[f] = xx.ravel()
            elif f == plot_y_feat:
                grid_data[f] = yy.ravel()
            else:
                grid_data[f] = fixed_values[f]

        grid_scaled = scaler.transform(grid_data)

        # 조정 전 경계선 (0.5)
        probs_b = log_reg_before.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
        fig.add_trace(
            go.Contour(
                z=probs_b,
                x=np.linspace(x_min, x_max, 100),
                y=np.linspace(y_min, y_max, 100),
                showscale=False,
                contours=dict(coloring="lines", showlines=True, start=0.5, end=0.5, size=0.1),
                line=dict(color="blue", width=2, dash="dash"),
                name="조정 전 경계선 (0.5)",
            )
        )

        # 조정 후 경계선 (0.5)
        probs_a = log_reg_after.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
        fig.add_trace(
            go.Contour(
                z=probs_a,
                x=np.linspace(x_min, x_max, 100),
                y=np.linspace(y_min, y_max, 100),
                showscale=False,
                contours=dict(coloring="lines", showlines=True, start=0.5, end=0.5, size=0.1),
                line=dict(color="red", width=2, dash="solid"),
                name="조정 후 경계선 (0.5)",
            )
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info("🔵 파란색 점선: 가중치 조정 전 경계선 | 🔴 빨간색 실선: 가중치 조정 후(balanced) 경계선")

else:
    st.markdown("속성을 3개 이상 선택하셨으므로 **3차원 산점도**와 **3차원 평면 경계선**을 표시합니다. 마우스로 회전하여 확인할 수 있습니다.")

    c1, c2, c3 = st.columns(3)
    with c1:
        p_x_kor = st.selectbox("X축 속성", options=selected_korean_features, index=0)
    with c2:
        p_y_kor = st.selectbox("Y축 속성", options=selected_korean_features, index=1)
    with c3:
        p_z_kor = st.selectbox("Z축 속성", options=selected_korean_features, index=2)

    p_x = reverse_feature_mapping[p_x_kor]
    p_y = reverse_feature_mapping[p_y_kor]
    p_z = reverse_feature_mapping[p_z_kor]

    other_features = [f for f in selected_features if f not in [p_x, p_y, p_z]]
    fixed_values = {f: df_train[f].median() for f in other_features}

    df_test_plot = df_test.copy()
    df_test_plot["실제_뇌졸중"] = df_test_plot["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

    fig_3d = px.scatter_3d(
        df_test_plot,
        x=p_x,
        y=p_y,
        z=p_z,
        color="실제_뇌졸중",
        labels={p_x: p_x_kor, p_y: p_y_kor, p_z: p_z_kor},
        title="3차원 산점도 및 가중치 조정 전후 결정 평면 비교",
        color_discrete_map={"정상 (0)": "#636EFA", "뇌졸중 (1)": "#EF553B"},
        opacity=0.7,
    )

    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown("💡 *3차원 공간 내에서 모델이 가중치를 조정함에 따라 뇌졸중 환자를 더 적극적으로 포착하도록 결정 영역이 이동합니다.*")
