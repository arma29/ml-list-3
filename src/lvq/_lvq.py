import numpy as np
from sklearn.model_selection import train_test_split

class LVQ():
    def __init__(self, prototypes_number=10, version='LVQ1'):
        self.version = version
        self.prototypes_number = prototypes_number
        self.__alpha = 0.1
        self.__window_width = 0.2
        self.__epslon = 0.1
        self.__epochs = 10
    
    def generate(self,X, y):
        if(self.version == 'None'):
            return (X,y)
        prototypes = self.__get_prototypes(X,y)

        if(self.version == 'LVQ1'):
            return self.__LVQ1(X,y,prototypes)
        elif(self.version == 'LVQ2.1'):
            return self.__LVQ2_1(X,y,self.__LVQ1(X,y,prototypes))
        elif(self.version == 'LVQ3'):
            return self.__LVQ3(X,y,self.__LVQ1(X,y,prototypes))
        raise Exception(
            'The \'version\' should be one of: LVQ1,LVQ2.1, LVQ3')

    def __get_prototypes(self, X, y):
        # Amostragem randômica estratificada
        proportion = self.prototypes_number / len(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=proportion, random_state=1, shuffle=True, stratify=y)
        # print(f'Tamanho do Teste: {len(X_test)}')
        return (X_test,y_test)

    def __LVQ1(self,X,y,prototypes):
        for _ in range(self.__epochs):
            # Para cada padrão do conjunto de treinamento x:
            for i in range(len(X)):
                # Encontre protótipo p mais próximo de x
                x = X[i]
                lst = sorted(enumerate(prototypes[0]),
                            key=lambda p: np.linalg.norm(x - p[1]))[0]
                index = lst[0]
                p = lst[1]

                if(prototypes[1][index] == y[i]):
                    # Se classe(p) = classe(x), aproxime p de x
                    # p = p + alpha(t) x (x - p)
                    p = np.add(p, self.__alpha*(x - p))

                else:
                    # Se classe(p) ≠ classe(x), afaste p de x
                    # p = p - alpha(t) x (x - p)
                    p = np.subtract(p, self.__alpha*(x - p))

                # Atualiza o protótipo
                prototypes[0][index] = p

        #Lembrando que a saída de LVQ1 serve de entrada para o LVQ2.1 e LVQ3
        return prototypes

    def __LVQ2_1(self,X,y,prototypes):
        for _ in range(self.__epochs):
            # Para cada padrão do conjunto de treinamento x:
            for i in range(len(X)):
                # Encontre os dois protótipos, pi e pj, mais próximos de x
                x = X[i]
                lst = sorted(enumerate(prototypes[0]),
                            key=lambda p: np.linalg.norm(x - p[1]))[:2]

                indexes = [x[0] for x in lst]
                pi = lst[0][1]
                pj = lst[1][1]

                if(self.__inside_window(x,pi,pj)):
                    if(prototypes[1][indexes[0]] != prototypes[1][indexes[1]]):
                        if(prototypes[1][indexes[0]] == y[i]):
                            # Aproxima pi ; Afasta pj
                            # pi = pi + alpha(t) x (x - pi)
                            # pj = pj - alpha(t) x (x - pj)
                            pi = np.add(pi, self.__alpha*(x - pi))
                            pj = np.subtract(pj, self.__alpha*(x - pj))

                        else:
                            # Aproxima pj ; Afasta pi
                            # pi = pi - alpha(t) x (x - pi)
                            # pj = pj + alpha(t) x (x - pj)
                            pi = np.subtract(pi, self.__alpha*(x - pi))
                            pj = np.add(pj, self.__alpha*(x - pj))
                
                prototypes[0][indexes[0]] = pi
                prototypes[0][indexes[1]] = pj

        return prototypes

    def __inside_window(self,x,pi,pj):
        s = (1 - self.__window_width) / (1 + self.__window_width)
        # Para não dar divisão por zero
        dist_xpi = np.linalg.norm(x - pi) + self.__epslon 
        dist_xpj = np.linalg.norm(x - pj) + self.__epslon
        
        return min(dist_xpi/dist_xpj , dist_xpj/dist_xpi) > s

    def __LVQ3(self,X,y,prototypes):
        for _ in range(self.__epochs):
            # Para cada padrão do conjunto de treinamento x:
            for i in range(len(X)):
                # Encontre os dois protótipos, pi e pj, mais próximos de x
                x = X[i]
                lst = sorted(enumerate(prototypes[0]),
                            key=lambda p: np.linalg.norm(x - p[1]))[:2]

                indexes = [x[0] for x in lst]
                pi = lst[0][1]
                pj = lst[1][1]

                if(self.__inside_window(x,pi,pj)):
                    if(prototypes[1][indexes[0]] != prototypes[1][indexes[1]]):
                        if(prototypes[1][indexes[0]] == y[i]):
                            # Aproxima pi ; Afasta pj
                            # pi = pi + alpha(t) x (x - pi)
                            # pj = pj - alpha(t) x (x - pj)
                            pi = np.add(pi, self.__alpha*(x - pi))
                            pj = np.subtract(pj, self.__alpha*(x - pj))

                        else:
                            # Aproxima pj ; Afasta pi
                            # pi = pi - alpha(t) x (x - pi)
                            # pj = pj + alpha(t) x (x - pj)
                            pi = np.subtract(pi, self.__alpha*(x - pi))
                            pj = np.add(pj, self.__alpha*(x - pj))
                    else:
                        # Caso tenham classes iguais, aproxima-se ambos
                        pi = np.add(pi, self.__epslon*self.__alpha*(x - pi))
                        pj = np.add(pj, self.__epslon*self.__alpha*(x - pj))

                prototypes[0][indexes[0]] = pi
                prototypes[0][indexes[1]] = pj

        return prototypes
