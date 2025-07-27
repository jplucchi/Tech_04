import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class ObesityPipeline:
    def __init__(self):
        self.numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        self.categorical_features = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        self.preprocessor = None
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path):
        """Carrega e faz limpeza inicial dos dados"""
        df = pd.read_csv(file_path)
        
        # Verificar valores ausentes
        print("Valores ausentes por coluna:")
        print(df.isnull().sum())
        
        # Estatísticas básicas
        print(f"\nShape do dataset: {df.shape}")
        print(f"Colunas: {list(df.columns)}")
        
        return df
    
    def exploratory_analysis(self, df):
        """Análise exploratória dos dados"""
        print("\nDistribuição da variável target (Obesity):")
        print(df['Obesity'].value_counts())
        
        print("\nEstatísticas descritivas das variáveis numéricas:")
        print(df[self.numerical_features].describe())
        
        print("\nDistribuição das variáveis categóricas:")
        for col in self.categorical_features:
            if col in df.columns:
                print(f"\n{col}:")
                print(df[col].value_counts())
    
    def prepare_data(self, df):
        """Prepara os dados para treinamento"""
        X = df.drop('Obesity', axis=1)
        y = df['Obesity']
        
        # Encoder para a variável target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test
    
    def create_preprocessor(self):
        """Cria o pipeline de pré-processamento"""
        # Pipeline para variáveis numéricas
        numerical_transformer = StandardScaler()
        
        # Pipeline para variáveis categóricas
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        # Combinar transformações
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        return self.preprocessor
    
    def train_models(self, X_train, y_train):
        """Treina diferentes modelos e compara performance"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"Treinando modelo: {name}")
            print('='*50)
            
            # Criar pipeline completo
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # Treinar modelo
            pipeline.fit(X_train, y_train)
            
            # Avaliar no conjunto de treino
            train_score = pipeline.score(X_train, y_train)
            print(f"Acurácia no treino: {train_score:.4f}")
            
            results[name] = {
                'pipeline': pipeline,
                'train_score': train_score
            }
        
        return results
    
    def evaluate_model(self, pipeline, X_test, y_test, model_name):
        """Avalia o modelo no conjunto de teste"""
        print(f"\n{'='*50}")
        print(f"Avaliação do modelo: {model_name}")
        print('='*50)
        
        # Predições
        y_pred = pipeline.predict(X_test)
        test_score = pipeline.score(X_test, y_test)
        
        print(f"Acurácia no teste: {test_score:.4f}")
        
        # Classification Report
        print("\nClassification Report:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Matriz de confusão
        print("\nMatriz de Confusão:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return test_score, y_pred
    
    def select_best_model(self, results, X_test, y_test):
        """Seleciona o melhor modelo baseado na performance"""
        best_score = 0
        best_model = None
        best_name = ""
        
        for name, result in results.items():
            score, _ = self.evaluate_model(result['pipeline'], X_test, y_test, name)
            
            if score > best_score:
                best_score = score
                best_model = result['pipeline']
                best_name = name
        
        print(f"\n{'='*50}")
        print(f"MELHOR MODELO: {best_name}")
        print(f"Acurácia: {best_score:.4f}")
        print('='*50)
        
        return best_model, best_name, best_score
    
    def save_model(self, model, model_name):
        """Salva o modelo treinado"""
        filename = f"obesity_model_{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, filename)
        
        # Salvar também o label encoder
        joblib.dump(self.label_encoder, "label_encoder.joblib")
        
        print(f"Modelo salvo como: {filename}")
        print("Label encoder salvo como: label_encoder.joblib")
        
        return filename

def main():
    # Inicializar pipeline
    pipeline = ObesityPipeline()
    
    # Carregar dados
    df = pipeline.load_data('Obesity.csv')
    
    # Análise exploratória
    pipeline.exploratory_analysis(df)
    
    # Preparar dados
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    
    # Criar preprocessor
    pipeline.create_preprocessor()
    
    # Treinar modelos
    results = pipeline.train_models(X_train, y_train)
    
    # Selecionar melhor modelo
    best_model, best_name, best_score = pipeline.select_best_model(results, X_test, y_test)
    
    # Salvar modelo
    if best_score >= 0.75:
        pipeline.save_model(best_model, best_name)
        print(f"\nModelo atende ao requisito de acurácia > 75%!")
    else:
        print(f"\nATENÇÃO: Modelo não atende ao requisito de acurácia > 75%")
        print("Considere ajustar hiperparâmetros ou tentar outros algoritmos.")

if __name__ == "__main__":
    main()