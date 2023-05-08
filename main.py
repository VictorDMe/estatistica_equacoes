import numpy as np
from scipy import stats as st
import streamlit as stream
import seaborn as sns
import matplotlib.pyplot as plt


class Estatistica:
    
    def __init__(self, dados:list[float]=[]) -> None:
        """Classe para ser utilizada nas aulas de estatística

        Args:
            dados (list[float], optional): Conjunto de dados. Necessário para uso de alguns métodos. Defaults to [].
        """        
        self.dados = dados
    
    def calculo_amostral(self, confianca:float, p:float, erro:float, N:int) -> int:
        """Retorna a quantidade de amostras totais.

        Args:
            confianca (float): Coeficiente da confiança da amostra;
            p (float): split das ocorrências, sendo comum o uso de p: 0,5 e 0,8;
            erro (float): erro da amostra;
            N (int): Quantidade da população

        Returns:
            int: Quantidade de amostra que deve ser coletada;
        """ 
        return round(((confianca**2 * p * (1 - p)) / erro**2) / (1 + ((confianca**2 * p * (1-p)) / (erro**2 * N))))

    def media(self, array:list[float]) -> float:
        """Retorna a média de um conjunto de dados.

        Args:
            array (list[float]): Conjunto de dados.

        Returns:
            float: Média dos valores.
        """        
        return np.mean(array)
        
    def mediana(self, array:list[float]) -> float:
        """Retorna a mediana de um conjunto de dados.

        Args:
            array (list[float]): Conjunto de dados.

        Returns:
            float: Mediana dos valores.
        """        
        return np.median(array)

    def moda(self, array:list[float]) -> float:
        """Retorna a moda de um conjunto de dados.

        Args:
            array (list[float]): conjunto de dados.

        Returns:
            float: moda.
        """   
        moda = st.mode(array, keepdims=False)
        return moda[0] if moda[1] > 1 else None

    def desvio_padrao(self, array:list[int]) -> float:
        '''
        Retorna o desvio padrão de um conjunto de dados
        '''
        return round(np.std(array), 4)

    def coef_pearson(self, array:list[float]) -> float:
        '''
        Retorna coeficiente de Pearson de um conjunto de dados
        '''
        return (3 * (self.media(array) - self.mediana(array))) / self.desvio_padrao(array)

    def curtose(self, array:list[int]) -> float:
        '''
        Retorna a curtose de um conjunto de dados
        '''
        return round(st.kurtosis(array, fisher=False), 3)

    def resumir(self, analise:bool=True) -> dict:
        """Resume os dados encontrados na variável 'dados' com os seguintes métodos:
        - Média, mediana, moda, desvio padrão, pearson e curtose.

        Raises:
            NameError: Se a variável 'dados' não existir.

        Returns:
            dict: resumo dos dados;
        """        
        if self.dados:
            pearson_value = round(self.coef_pearson(self.dados), 3)
            curtose_value = self.curtose(self.dados)

            if analise:
                # Analisando valor de pearson
                # ASSIMETRIA
                if pearson_value == 0:
                    pearson_value = f'{pearson_value} - Simétrica'
                elif pearson_value > 0:
                    pearson_value = f'{pearson_value} - Assimétrica à direita ou positiva - ' \
                    f'{"MODERADA" if 0.15 < abs(pearson_value) < 1 else "FORTE"}'
                else:
                    pearson_value = f'{pearson_value} - Assimétrica à esquerda ou negativa - ' \
                    f'{"MODERADA" if 0.15 < abs(pearson_value) < 1 else "FORTE"}'
                
                # Analisando valor da curtose
                if curtose_value == 0.263:
                    curtose_value = f'{curtose_value} - Mesocúrtica'
                elif curtose_value > 0.263:
                    curtose_value = f'{curtose_value} - Platicúrtica'
                else:
                    curtose_value = f'{curtose_value} - Leptocúrtica'

            return {
                'media': self.media(self.dados),
                'mediana': self.mediana(self.dados),
                'moda': self.moda(self.dados),
                'desvio_padrao': self.desvio_padrao(self.dados),
                'pearson': pearson_value,
                'curtose': curtose_value
            }
        else:
            raise NameError('Variável "dados" não foi criada')

    def plotar(self, streamlit=False):
        '''
        Somente executará se a variável 'dados' existir.
        Plota dois gráficos:
            1 - KDE (Densidade do Kernel), que, tem como objetivo
        aferir a probabilidade de um acontecimento.
            2 - Histograma, que, tem como objetivo descobrir a quantidade de
            ocorrências de determinado acontecimento.
        '''

        if self.dados:
            sns.set_theme()
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle("DISTRIBUIÇÃO DOS DADOS")

            # DENSIDADE POR KERNEL
            sns.kdeplot(self.dados, ax=axs[0])
            axs[0].set_ylabel('Probabilidade')
            axs[0].set_title('Densidade por Kernel')

            # HISTOGRAMA
            sns.histplot(self.dados, ax=axs[1])
            axs[1].set_ylabel('Ocorrências')
            axs[1].set_title('Histograma')
            
            # BOX PLOT - IDENTIFICAÇÃO DE OUTLIERS
            sns.boxplot(self.dados, ax=axs[2])
            axs[2].set_ylabel('Valores encontrados')
            axs[2].set_xticks([])
            axs[2].set_title('Boxplot')
            
            if not streamlit:
                plt.show()
            else:
                stream.pyplot(plt)

        else:
            raise NameError('Variável "dados" não foi criada')

def enviar(data):
    data.strip().replace(' ', '')
    data = [float(z) for z in data.split(',')]
    obj = Estatistica(dados=data)

    stream.header('Análise')
    obj.plotar(streamlit=True)

    a = obj.resumir()
    stream.write(f"*Média:* {a['media']}")
    stream.write(f"*Mediana:* {a['mediana']}")
    stream.write(f"*Moda:* {a['moda']}")
    stream.write(f"*Desvio padrão:* {a['desvio_padrao']}")
    stream.write(f"*Pearson:* {a['pearson']}")
    stream.write(f"*Curtose:* {a['curtose']}")

if __name__ == '__main__':
    stream.set_page_config(page_title='Estatística',
                           page_icon='🎲')

    stream.title('Estatística')
    nums = []
    nums = stream.text_input('Digite os valores separados por vírgula')
    
    bt = stream.button('ENVIAR')

    if bt:
        enviar(data=nums)