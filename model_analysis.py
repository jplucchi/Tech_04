import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelAnalyzer:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        
    def load_model(self):
        """Carrega o modelo treinado"""
        try:
            self.model = joblib.load("obesity_model_gradient_boosting.joblib")
            self.label_encoder = joblib.load("label_encoder.joblib")
            print("Modelo carregado com sucesso!")
            return True
        except FileNotFoundError:
            print("Modelo não encontrado. Execute primeiro o pipeline de treinamento.")
            return False
    
    def analyze_feature_importance(self, X_train, feature_names):
        """Analisa a importância das features"""
        if self.model is None:
            print("Modelo não carregado.")
            return
        
        # Obter importâncias do modelo
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            importances = self.model.named_steps['classifier'].feature_importances_
            
            # Criar DataFrame com importâncias
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Features mais importantes:")
            print(importance_df.head(10))
            
            # Plotar importâncias
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importância')
            plt.title('Top 15 Features por Importância')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            print("Modelo não suporta feature importance.")
    
    def cross_validation_analysis(self, X, y, cv=5):
        """Realiza validação cruzada"""
        if self.model is None:
            print("Modelo não carregado.")
            return
        
        print(f"Realizando validação cruzada com {cv} folds...")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"Acurácia média: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        print(f"Scores individuais: {scores}")
        
        return scores
    
    def analyze_predictions(self, X_test, y_test):
        """Analisa as predições do modelo"""
        if self.model is None:
            print("Modelo não carregado.")
            return
        
        # Fazer predições
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Classification Report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # Matriz de Confusão
        print("\nMatriz de Confusão:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Plotar matriz de confusão
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Matriz de Confusão')
        plt.ylabel('Valor Real')
        plt.xlabel('Predição')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return y_pred, y_pred_proba
    
    def analyze_misclassifications(self, X_test, y_test):
        """Analisa casos de classificação incorreta"""
        if self.model is None:
            print("Modelo não carregado.")
            return
        
        y_pred = self.model.predict(X_test)
        
        # Encontrar classificações incorretas
        incorrect_mask = y_test != y_pred
        incorrect_indices = np.where(incorrect_mask)[0]
        
        print(f"Total de classificações incorretas: {len(incorrect_indices)}")
        print(f"Porcentagem de erro: {len(incorrect_indices)/len(y_test)*100:.2f}%")
        
        if len(incorrect_indices) > 0:
            print("\nPrimeiros 5 casos de classificação incorreta:")
            for i in incorrect_indices[:5]:
                real_class = self.label_encoder.inverse_transform([y_test.iloc[i]])[0]
                pred_class = self.label_encoder.inverse_transform([y_pred[i]])[0]
                print(f"Índice {i}: Real={real_class}, Predito={pred_class}")
        
        return incorrect_indices

def main():
    """Função principal para análise do modelo"""
    analyzer = ModelAnalyzer()
    
    # Carregar modelo
    if not analyzer.load_model():
        return
    
    # Carregar dados para análise
    df = pd.read_csv('Obesity.csv')
    
    # Preparar dados (mesmo processo do pipeline)
    X = df.drop('Obesity', axis=1)
    y = df['Obesity']
    y_encoded = analyzer.label_encoder.transform(y)
    
    # Dividir dados
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Obter nomes das features após preprocessamento
    # Isso é uma aproximação - no pipeline real você salvaria os nomes
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    
    # Para análise, vamos usar nomes simplificados
    feature_names = numerical_features + [f"{cat}_encoded" for cat in categorical_features]
    
    print("="*60)
    print("ANÁLISE DETALHADA DO MODELO")
    print("="*60)
    
    # Análise de importância das features
    analyzer.analyze_feature_importance(X_train, feature_names)
    
    # Validação cruzada
    print("\n" + "="*60)
    print("VALIDAÇÃO CRUZADA")
    print("="*60)
    analyzer.cross_validation_analysis(X_train, y_train)
    
    # Análise das predições
    print("\n" + "="*60)
    print("ANÁLISE DAS PREDIÇÕES")
    print("="*60)
    analyzer.analyze_predictions(X_test, y_test)
    
    # Análise de classificações incorretas
    print("\n" + "="*60)
    print("ANÁLISE DE ERROS")
    print("="*60)
    analyzer.analyze_misclassifications(X_test, y_test)

if __name__ == "__main__":
    main()