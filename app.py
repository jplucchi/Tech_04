import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(
    page_title="Sistema de Predição de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Carrega o modelo treinado"""
    try:
        import joblib
        model = joblib.load("obesity_model_gradient_boosting.joblib")
        label_encoder = joblib.load("label_encoder.joblib")
        return model, label_encoder
    except FileNotFoundError:
        st.error("❌ Modelo não encontrado. Execute primeiro: `python data_pipeline.py`")
        return None, None
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        st.info("💡 Tente executar: `pip install --upgrade numpy scikit-learn joblib`")
        return None, None

@st.cache_data
def load_data():
    """Carrega os dados para análise"""
    try:
        return pd.read_csv("Obesity.csv")
    except FileNotFoundError:
        st.error("❌ Dataset 'Obesity.csv' não encontrado.")
        return None

def create_prediction_interface():
    """Interface para predição individual"""
    st.markdown("<h2 style='text-align: center;'>Sistema de Predição de Obesidade</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Insira os dados do paciente para obter uma predição do nível de obesidade:</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Dados Demográficos")
        gender = st.selectbox("Gênero", ["Male", "Female"])
        age = st.number_input("Idade", min_value=10, max_value=100, value=25)
        height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
        
        st.markdown("### 🧬 Histórico Familiar")
        family_history = st.selectbox("Histórico familiar de obesidade", ["yes", "no"])
        
        # Preview do IMC em tempo real
        imc_preview = weight / (height ** 2)
        if imc_preview < 18.5:
            imc_status = "Abaixo do peso"
            color = "#4169E1"
        elif imc_preview < 25:
            imc_status = "Peso normal" 
            color = "#32CD32"
        elif imc_preview < 30:
            imc_status = "Sobrepeso"
            color = "#FFD700"
        else:
            imc_status = "Obesidade"
            color = "#FF6347"
            
        st.markdown(f"""
        <div style='background-color: {color}20; padding: 15px; border-radius: 8px; border-left: 4px solid {color}; margin: 10px 0;'>
            <h4 style='color: {color}; margin: 0;'>📊 IMC Preview</h4>
            <p style='margin: 5px 0; font-size: 20px; font-weight: bold; color: {color};'>{imc_preview:.1f}</p>
            <p style='margin: 0; color: {color};'>{imc_status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Guia de referência IMC
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; margin: 10px 0;'>
            <h4 style='color: #495057; margin: 0 0 10px 0;'>📋 Referência IMC</h4>
            <p style='margin: 2px 0; color: #4169E1;'><strong>< 18.5:</strong> Abaixo do peso</p>
            <p style='margin: 2px 0; color: #32CD32;'><strong>18.5-24.9:</strong> Peso normal</p>
            <p style='margin: 2px 0; color: #FFD700;'><strong>25.0-29.9:</strong> Sobrepeso</p>
            <p style='margin: 2px 0; color: #FF6347;'><strong>≥ 30.0:</strong> Obesidade</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🍽️ Hábitos Alimentares")
        favc = st.selectbox("Come alimentos altamente calóricos frequentemente", ["yes", "no"])
        fcvc = st.slider("Frequência de consumo de vegetais", 1.0, 3.0, 2.0, step=0.1)
        ncp = st.slider("Número de refeições principais por dia", 1.0, 4.0, 3.0, step=0.1)
        caec = st.selectbox("Come entre as refeições", ["no", "Sometimes", "Frequently", "Always"])
        
        # Indicador de hábitos alimentares
        score_alimentar = 0
        if favc == "no":
            score_alimentar += 25
        if fcvc >= 2.5:
            score_alimentar += 25
        if ncp >= 3:
            score_alimentar += 25
        if caec in ["no", "Sometimes"]:
            score_alimentar += 25
            
        if score_alimentar >= 75:
            status_alimentar = "Excelente"
            cor_alimentar = "#32CD32"
        elif score_alimentar >= 50:
            status_alimentar = "Bom"
            cor_alimentar = "#FFD700"
        else:
            status_alimentar = "Precisa melhorar"
            cor_alimentar = "#FF6347"
            
        st.markdown(f"""
        <div style='background-color: {cor_alimentar}20; padding: 15px; border-radius: 8px; border-left: 4px solid {cor_alimentar}; margin: 10px 0;'>
            <h4 style='color: {cor_alimentar}; margin: 0;'>🍎 Avaliação Alimentar</h4>
            <p style='margin: 5px 0; font-size: 18px; font-weight: bold; color: {cor_alimentar};'>{status_alimentar}</p>
            <p style='margin: 0; color: {cor_alimentar}; font-size: 14px;'>Score: {score_alimentar}/100</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🏃‍♂️ Estilo de Vida")
        smoke = st.selectbox("Fuma", ["yes", "no"])
        ch2o = st.slider("Consumo de água (litros/dia)", 1.0, 3.0, 2.0, step=0.1)
        scc = st.selectbox("Monitora calorias", ["yes", "no"])
        faf = st.slider("Frequência de atividade física (0-3)", 0.0, 3.0, 1.0, step=0.1)
        tue = st.slider("Tempo usando dispositivos tecnológicos (horas)", 0.0, 2.0, 1.0, step=0.1)
        calc = st.selectbox("Frequência de consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Meio de transporte", ["Public_Transportation", "Walking", "Automobile", "Bike", "Motorbike"])
        
        # Indicador de estilo de vida
        score_vida = 0
        if smoke == "no":
            score_vida += 20
        if ch2o >= 2.0:
            score_vida += 20
        if faf >= 2.0:
            score_vida += 20
        if tue <= 1.0:
            score_vida += 20
        if calc in ["no", "Sometimes"]:
            score_vida += 20
            
        if score_vida >= 80:
            status_vida = "Muito saudável"
            cor_vida = "#32CD32"
        elif score_vida >= 60:
            status_vida = "Saudável"
            cor_vida = "#FFD700"
        else:
            status_vida = "Requer atenção"
            cor_vida = "#FF6347"
            
        st.markdown(f"""
        <div style='background-color: {cor_vida}20; padding: 15px; border-radius: 8px; border-left: 4px solid {cor_vida}; margin: 10px 0;'>
            <h4 style='color: {cor_vida}; margin: 0;'>💪 Estilo de Vida</h4>
            <p style='margin: 5px 0; font-size: 18px; font-weight: bold; color: {cor_vida};'>{status_vida}</p>
            <p style='margin: 0; color: {cor_vida}; font-size: 14px;'>Score: {score_vida}/100</p>
        </div>
        """, unsafe_allow_html=True)

    # Botão de predição centralizado e maior
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col_left, col_center, col_right = st.columns([1, 1, 1])
    
    with col_center:
        predict_button = st.button(
            "🔬 REALIZAR PREDIÇÃO", 
            type="primary",
            use_container_width=True,
            help="Clique para obter a predição do nível de obesidade"
        )
    
    # CSS customizado para o botão
    st.markdown("""
    <style>
    .stButton > button {
        height: 60px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid #FF6B6B;
        background: linear-gradient(45deg, #FF6B6B, #FF8E8E);
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #FF8E8E, #FF6B6B);
    }
    </style>
    """, unsafe_allow_html=True)
    
    if predict_button:
        model, label_encoder = load_model()
        
        if model is not None and label_encoder is not None:
            try:
                # Criar DataFrame com os dados de entrada
                input_data = pd.DataFrame({
                    'Gender': [gender],
                    'Age': [age],
                    'Height': [height],
                    'Weight': [weight],
                    'family_history': [family_history],
                    'FAVC': [favc],
                    'FCVC': [fcvc],
                    'NCP': [ncp],
                    'CAEC': [caec],
                    'SMOKE': [smoke],
                    'CH2O': [ch2o],
                    'SCC': [scc],
                    'FAF': [faf],
                    'TUE': [tue],
                    'CALC': [calc],
                    'MTRANS': [mtrans]
                })
                
                # Fazer predição
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                
                # Decodificar resultado
                predicted_class = label_encoder.inverse_transform([prediction])[0]
                
                # Exibir resultado
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align: center; color: #2E8B57; background-color: #F0FFF0; padding: 20px; border-radius: 10px; border: 2px solid #2E8B57;'>🎯 Resultado da Predição: <strong>{predicted_class}</strong></h2>", unsafe_allow_html=True)
                
                # Mostrar probabilidades
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>📊 Probabilidades por Classe</h3>", unsafe_allow_html=True)
                
                prob_df = pd.DataFrame({
                    'Nível de Obesidade': label_encoder.classes_,
                    'Probabilidade': probabilities
                }).sort_values('Probabilidade', ascending=False)
                
                fig = px.bar(prob_df, x='Nível de Obesidade', y='Probabilidade',
                            title="<b>Distribuição de Probabilidades</b>",
                            color='Probabilidade',
                            color_continuous_scale='viridis')
                fig.update_layout(title_x=0.5, height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Card do IMC
                st.markdown(f"""
                <div style='text-align: center; background-color: {color}20; padding: 20px; border-radius: 10px; border: 2px solid {color}; margin: 20px 0;'>
                    <h3 style='color: {color}; margin: 0;'>📏 Índice de Massa Corporal (IMC)</h3>
                    <h2 style='color: {color}; margin: 10px 0;'>{imc_preview:.2f}</h2>
                    <p style='color: {color}; margin: 0; font-weight: bold;'>{imc_status}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Erro durante a predição: {str(e)}")
                st.info("💡 Verifique se o modelo foi treinado corretamente executando: `python data_pipeline.py`")
        else:
            st.warning("⚠️ Modelo não carregado. Execute primeiro o pipeline de treinamento.")

def create_analytics_dashboard():
    """Dashboard analítico com insights"""
    st.markdown("<h2 style='text-align: center;'>Dashboard Analítico - Insights sobre Obesidade</h2>", unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        st.error("❌ Não foi possível carregar o dataset.")
        return
    
    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Pacientes", len(df))
    
    with col2:
        avg_age = df['Age'].mean()
        st.metric("Idade Média", f"{avg_age:.1f} anos")
    
    with col3:
        obesity_count = df['Obesity'].str.contains('Obesity').sum() if 'Obesity' in df.columns else 0
        obesity_rate = (obesity_count / len(df)) * 100
        st.metric("Taxa de Obesidade", f"{obesity_rate:.1f}%")
    
    with col4:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        avg_bmi = df['BMI'].mean()
        st.metric("IMC Médio", f"{avg_bmi:.1f}")
    
    # Gráficos em abas
    tab1, tab2, tab3 = st.tabs(["📊 Distribuições", "🔍 Comparações", "🔗 Correlações"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            obesity_counts = df['Obesity'].value_counts()
            fig1 = px.pie(values=obesity_counts.values, names=obesity_counts.index, 
                         title="<b>Distribuição dos Níveis de Obesidade</b>")
            fig1.update_layout(title_x=0.5)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            gender_counts = df['Gender'].value_counts()
            fig2 = px.bar(x=gender_counts.index, y=gender_counts.values,
                         title="<b>Distribuição por Gênero</b>")
            fig2.update_layout(xaxis_title="Gênero", yaxis_title="Contagem", title_x=0.5)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                gender_obesity = df.groupby(['Gender', 'Obesity']).size().reset_index(name='Count')
                fig3 = px.bar(gender_obesity, x='Gender', y='Count', color='Obesity',
                             title="<b>Distribuição de Obesidade por Gênero</b>")
                fig3.update_layout(title_x=0.5)
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao criar gráfico: {e}")
        
        with col2:
            try:
                fig4 = px.box(df, x='Obesity', y='Age', 
                             title="<b>Distribuição de Idade por Nível de Obesidade</b>")
                fig4.update_layout(xaxis_tickangle=45, title_x=0.5)
                st.plotly_chart(fig4, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao criar gráfico: {e}")
    
    with tab3:
        st.markdown("<h3 style='text-align: center;'>Correlações entre Variáveis Numéricas</h3>", unsafe_allow_html=True)
        try:
            numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']
            available_cols = [col for col in numerical_cols if col in df.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = df[available_cols].corr()
                fig_corr = px.imshow(corr_matrix, 
                                   text_auto=True, 
                                   aspect="auto",
                                   title="<b>Matriz de Correlação</b>",
                                   color_continuous_scale='RdBu')
                fig_corr.update_layout(title_x=0.5)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Dados insuficientes para matriz de correlação")
        except Exception as e:
            st.error(f"Erro ao criar matriz de correlação: {e}")
    
    # Insights principais
    st.markdown("<h3 style='text-align: center;'>Principais Insights para Equipe Médica</h3>", unsafe_allow_html=True)
    
    insights = [
        "📊 **Distribuição populacional**: Análise dos diferentes níveis de obesidade na amostra",
        "⚖️ **Fatores de risco**: Correlação entre hábitos alimentares e níveis de obesidade", 
        "🏃‍♂️ **Atividade física**: Impacto da frequência de exercícios na prevenção da obesidade",
        "🍽️ **Padrões alimentares**: Relação entre consumo de vegetais e controle de peso",
        "🚗 **Estilo de vida**: Influência do meio de transporte no nível de atividade física"
    ]
    
    for insight in insights:
        st.write(insight)

def main():
    """Função principal da aplicação"""
    st.markdown("<h1 style='text-align: center;'>Sistema Inteligente de Predição de Obesidade</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><strong>Desenvolvido por:</strong> João Paulo Lucchi | <strong>RM:</strong> 360649 | <strong>Tech Challenge - Fase 04</strong></p>", unsafe_allow_html=True)
    
    # Sidebar para navegação
    st.sidebar.title("Navegação")
    page = st.sidebar.selectbox("Escolha uma página:", 
                               ["Predição Individual", "Dashboard Analítico"])
    
    if page == "Predição Individual":
        create_prediction_interface()
    elif page == "Dashboard Analítico":
        create_analytics_dashboard()
    
    # Informações adicionais na sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sobre o Sistema")
    st.sidebar.info(
        "Este sistema utiliza Machine Learning para auxiliar profissionais "
        "de saúde na avaliação e predição de níveis de obesidade em pacientes."
    )
    
    st.sidebar.markdown("### Tecnologias Utilizadas")
    st.sidebar.write("• Python & Scikit-learn")
    st.sidebar.write("• Streamlit")
    st.sidebar.write("• Plotly")
    st.sidebar.write("• Pandas & NumPy")
    
    # Status do sistema
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Status do Sistema")
    model, label_encoder = load_model()
    if model is not None:
        st.sidebar.success("✅ Modelo carregado")
    else:
        st.sidebar.error("❌ Modelo não encontrado")
        st.sidebar.info("Execute: `python data_pipeline.py`")

if __name__ == "__main__":
    main()