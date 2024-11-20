import numpy as np
import pandas as pd

def CalcularEntropia(data):
    classes = data.value_counts(normalize=True)
    entropia = -np.sum(classes * np.log2(classes))
    return entropia

def CalculGuanyInformacio(data, atribut, columnaObjectiu):
    entropiaTotal = CalcularEntropia(data[columnaObjectiu])
    
    valorsAtribut = data[atribut].unique()
    
    entropia = 0
    for valor in valorsAtribut:
        subset = data[data[atribut] == valor]
        entropia += (len(subset) / len(data)) * CalcularEntropia(subset[columnaObjectiu])
    
    return entropiaTotal - entropia

def BuscarMillorAtribut(data, columnaObjectiu):

    millorGuany = -1  
    millorAtribut = None 

    for atribut in data.columns:
        if atribut != columnaObjectiu:
            guanyInformacio = CalculGuanyInformacio(data, atribut, columnaObjectiu)
            if guanyInformacio > millorGuany:
                millorGuany = guanyInformacio
                millorAtribut = atribut
    
    return millorAtribut

def ArbreDecisioID3(data, columnaObjectiu):

    # Si la columna objectiu nomes conte 1 valor retornem fulla
    if len(data[columnaObjectiu].unique()) == 1:
        return data[columnaObjectiu].iloc[0]
    
    # Si nomes ens queda la columna objectiu retornem el valor mes comu
    if len(data.columns) == 1: 
        return data[columnaObjectiu].mode()[0]
    
    #Busquem millor atribut
    millorAtribut = BuscarMillorAtribut(data, columnaObjectiu)
    
    # Crearem l'arbre en forma de diccionaris dins de diccionaris
    arbreDecisio = {millorAtribut: {}}
    
    #Separem el arbre en tans fills com valors unics tingui la columna
    valorsAtribut = data[millorAtribut].unique()
    for value in valorsAtribut:
        subset = data[data[millorAtribut] == value]
        # Construir subarbres de forma recursiva descartant la columna del millor atribut
        arbreDecisio[millorAtribut][value] = ArbreDecisioID3(subset.drop(columns=[millorAtribut]), columnaObjectiu)
    
    return arbreDecisio

def print_tree(arbre, level=0):
    
    if isinstance(arbre, dict):
        for atribut, branques in arbre.items():
            print(" " * level * 4 + f"{atribut}?")
            for valorBranca, subAbre in branques.items():
                print(" " * (level + 1) * 4 + f"({valorBranca})")
                print_tree(subAbre, level + 2)
    else:
        print(" " * level * 4 + f"-> {arbre}")

def main():
    dataset_csv = pd.read_csv("./bolets.csv")
    dataset = pd.DataFrame(dataset_csv)

    columnaObjectiu = 'class'
    arbreDecisio = ArbreDecisioID3(dataset, columnaObjectiu)
    print_tree(arbreDecisio)

if __name__ == '__main__':
    main()
    
