#Universidade Tecnológica Federal do Paraná - Campus Campoo Mourão
#Acadêmicos:Alexandre de Oliveira Júnior
#			Edson Vicente Junior
#Professor: Diego Bertolini
#Disciplina: Inteligência Artificial

#Algoritmo KNN

#Definição das bibliotecas
import csv
import random
import operator
import math

#Função para geração de uma matriz nula, que será utilizada para obtenção da matriz de confusão
def gerarMatriz (linhas, colunas):
	matriz = []
	for _ in range(linhas):
		matriz.append([0]*colunas)
	return matriz

#Função de abertura e vetorização dos arquivos
def loadDataset(filename, split, datatrain=[]):
	with open(filename, 'r') as f:
		linhas = csv.reader(f)
		dataset=list(linhas)
		for x in range(len(dataset)-1):
			for y in range(132):
				dataset[x][y]=float(dataset[x][y])
			if random.random()<split:
				datatrain.append(dataset[x])

#Função de normalização Min-max
def normalizeData(data):
	minimo = 1
	maximo = 0
	for x in range(len(data)-1):
		for y in range(132):
			if data[x][y]<minimo:
				minimo = data[x][y]
			if data[x][y]>maximo:
				maximo = data[x][y]
	for x in range(len(data)-1):
		for y in range(132):
			data[x][y]=float((data[x][y]-minimo)/(maximo-minimo))

#Função para o calculo da distância Euclediana
def euclideanDistance(inst1, inst2, length):
	distancia=0
	for x in range(length):
		distancia += pow((float(inst1[x])-float(inst2[x])), 2)
	return math.sqrt(distancia)

#Funçao para obtenção dos K vizinhos(amostras) mais proximas
def getNeighbors(datatrain, atualInst, k):
	distancias=[]
	comp = len(atualInst)-1
	for x in range(len(datatrain)):
		dist = euclideanDistance(atualInst, datatrain[x],comp)
		distancias.append((datatrain[x], dist))
	distancias.sort(key=operator.itemgetter(1))
	vizinhos=[]
	for x in range(k):
		vizinhos.append(distancias[x][0])
	return vizinhos

#Função para determinação do algarismo em análise
def getResponde(vizinhos):
	classVotos = {}
	for x in range(len(vizinhos)):
		resposta = vizinhos[x][-1]
		if resposta in classVotos:
			classVotos[resposta] += 1
		else:
			classVotos[resposta] = 1
	sortedVotos = sorted(classVotos.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotos[0][0]

#Função para determinação da acurácia do algoritmo
def getAccuracy(datatest, predict):
	correto = 0
	for x in range(len(datatest)):
		if datatest[x][-1] == predict[x]:
			correto += 1
	return (correto/float(len(datatest)))*100.0

#Função principal
def main():
	datatrain=[]
	datatest=[]

	#Possibilita indicar a porcentagem do banco de treinamento e de teste a ser utilizada, sendo que ao
	#remover dados, estes serão escolhidos aleatoriamente
	splitTreino = 1
	#splitTreino = 0.5
	#splitTreino = 0.25
	splitTeste=1
	loadDataset('treinamento.txt', splitTreino, datatrain)
	loadDataset('teste.txt', splitTeste, datatest)

	#Chamada de funçãp para normalização das amostras pelo método de Min-Max
	normalizeData(datatrain)
	normalizeData(datatest)

	print("Total de amostras do treino: " +repr(len(datatrain)))
	print("Total do amostras de teste: " +repr(len(datatest)))

	#Determina o número de K vizinhos a serem analizados
	k = 1

	predict = []
	matrizC = gerarMatriz(10,10)
	aux1 = 0
	aux2 = 0
	for x in range(len(datatest)):
		vizinhos = getNeighbors(datatrain, datatest[x], k)
		resultado = getResponde(vizinhos)
		predict.append(resultado)
		aux1 = int(resultado)
		aux2 = int(datatest[x][-1])
		matrizC[aux1][aux2] += 1
		#print("Predicao: " +repr(resultado)+ " Valor real: " +repr(datatest[x][-1]))
	acuracia = getAccuracy(datatest, predict)
	print("Acuracia: " +repr(acuracia)+ "%")

	print("Matriz de confusão: ")
	print(matrizC)

if __name__=='__main__':
	main()
