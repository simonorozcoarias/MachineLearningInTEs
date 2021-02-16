#!/bin/env pyhton

import sys
import os
import Bio
from Bio import SeqIO
import itertools
import time
import multiprocessing

"""
This function takes as input classified TEs at lineage level and transforms nucleotides into numbers
using different coding schemes such as DAX, EIIP, complementary, orthogonal, enthalpy, Galois4.
NOTE: enthalpy, Galois4 were calculated taking two nucleotides with slide windows of 2 nucleotides.
"""
def filterDBLineages(file,schema,maxlength):
	result = open(file+'.'+schema, "w")
	cont = 0
	print("doing: "+schema)
	if schema == 'DAX':
		labels = ['B'+str(x) for x in range(0,maxlength)]
		result.write('Label,'+','.join(labels)+'\n')
	elif schema == 'EIIP':
		labels = ['B'+str(x) for x in range(0,maxlength)]
		result.write('Label,'+','.join(labels)+'\n')
	elif schema == 'complementary':
		labels = ['B'+str(x) for x in range(0,maxlength)]
		result.write('Label,'+','.join(labels)+'\n')
	elif schema == 'orthogonal':
		labels = ['B'+str(x) for x in range(0,maxlength*4)]
		result.write('Label,'+','.join(labels)+'\n')
	elif schema == 'enthalpy':
		labels = ['B'+str(x) for x in range(0,int(maxlength/2))]
		result.write('Label,'+','.join(labels)+'\n')
	elif schema == 'Galois4':
		labels = ['B'+str(x) for x in range(0,int(maxlength/2))]
		result.write('Label,'+','.join(labels)+'\n')
	for te in SeqIO.parse(file, "fasta"):
		#print(cont)
		if len(te.seq) <= maxlength:
			caracters = te.seq
			# looking for label
			order = -1
			if str(te.id).upper().find("ALE-") != -1 or str(te.id).upper().find("RETROFIT-") != -1:
				order = 1
			# elif str(te.id).upper().find("ALESIA-") != -1:
			# 	order = 2
			elif str(te.id).upper().find("ANGELA-") != -1:
				order = 3
			elif str(te.id).upper().find("BIANCA-") != -1:
				order = 4
			# elif str(te.id).upper().find("BRYCO-") != -1:
			# 	order = 5
			# elif str(te.id).upper().find("LYCO-") != -1:
			# 	order = 6
			# elif str(te.id).upper().find("GYMCO-") != -1: 
			# 	order = 7
			elif str(te.id).upper().find("IKEROS-") != -1:
				order = 8
			elif str(te.id).upper().find("IVANA-") != -1 or str(te.id).upper().find("ORYCO-") != -1:
				order = 9
			# elif str(te.id).upper().find("OSSER-") != -1:
			# 	order = 10
			# elif str(te.id).upper().find("TAR-") != -1:
			# 	order = 11
			elif str(te.id).upper().find("TORK-") != -1:
				order = 12
			elif str(te.id).upper().find("SIRE-") != -1:
				order = 13
			elif str(te.id).upper().find("CRM-") != -1:
				order = 14
			# elif str(te.id).upper().find("CHLAMYVIR-") != -1:
			# 	order = 15
			elif str(te.id).upper().find("GALADRIEL-") != -1:
				order = 16
			elif str(te.id).upper().find("REINA-") != -1:
				order = 17
			elif str(te.id).upper().find("TEKAY-") != -1 or str(te.id).upper().find("DEL-") != -1:
				order = 18
			elif str(te.id).upper().find("ATHILA-") != -1:
				order = 19
			elif str(te.id).upper().find("TAT-") != -1:
				order = 20
			elif str(te.id).upper().find("OGRE-") != -1:
				order = 21
			elif str(te.id).upper().find("RETAND-") != -1:
				order = 22
			# elif str(te.id).upper().find("PHYGY-") != -1:
			# 	order = 23
			# elif str(te.id).upper().find("SELGY-") != -1:
			# 	order = 24
			#print(order) 
			if order != -1:
				if len(caracters) < maxlength:
					# to complete TEs with self-replication method
					times = int((maxlength-len(caracters))/len(caracters))+1
					caracters = str(caracters+(str(caracters)*(times+1)))[0:maxlength]
					# to complete TEs with NNs-filling method
					#diff = maxlength - len(caracters)
					#caracters += ''.join(['N' for x in range(diff)])
				if schema == 'DAX':
					newseq = ','.join([str(DAX(caracters[x])) for x in range(0, len(caracters))])
				elif schema == 'EIIP':
					newseq = ','.join([str(EIIP(caracters[x])) for x in range(0, len(caracters))])
				elif schema == 'complementary':
					newseq = ','.join([str(complementary(caracters[x])) for x in range(0, len(caracters))])
				elif schema == 'orthogonal':
					newseq = ','.join([str(orthogonal(caracters[x])) for x in range(0, len(caracters))])
				elif schema == 'enthalpy':
					newseq = ','.join([str(enthalpy(caracters[x:x+2])) for x in range(0, len(caracters), 2)])
				elif schema == 'Galois4':
					newseq = ','.join([str(Galois4(''.join(caracters[x:x+2]))) for x in range(0, len(caracters), 2)])
				result.write(str(order)+","+newseq+"\n")
			cont += 1
	result.close()

def filterSeqs(file,maxlength):
	result = open(file+'.superfamilies', "w")
	labels = ['B'+str(x) for x in range(0,maxlength)]
	result.write('Label,'+','.join(labels)+'\n')
	cont = 0
	for te in SeqIO.parse(file, "fasta"):
		print(cont)
		if len(te.seq) <= maxlength:
			caracters = te.seq
			# looking for label
			order = -1
			if te.id.find("RLC") != -1:
				order = 1
			elif te.id.find("RLG") != -1:
				order = 2
			print(order)
			if order != -1:
				if len(caracters) < maxlength:
					result.write(str(order)+","+str(caracters)+"\n")
			cont += 1
	result.close()

def DAX(nucl):
	conv = 4
	if nucl.upper() == 'A': #0.1
		conv = 2
	elif nucl.upper() == 'C': #0.9
		conv = 0
	elif nucl.upper() == 'G': #0.8
		conv = 3
	elif nucl.upper() == 'T': #0.2
		conv = 1
	return conv

def EIIP(nucl):
	conv = 0
	if nucl.upper() == 'A': #0.1
		conv = 0.1260
	elif nucl.upper() == 'C': #0.9
		conv = 0.1340
	elif nucl.upper() == 'G': #0.8
		conv = 0.0806
	elif nucl.upper() == 'T': #0.2
		conv = 0.1335
	return conv

def complementary(nucl):
	conv = 0
	if nucl.upper() == 'A':  #0.1
		conv = 2
	elif nucl.upper() == 'C': #0.9
		conv = -1
	elif nucl.upper() == 'G': #0.8
		conv = 1
	elif nucl.upper() == 'T': #0.2
		conv = -2
	return conv

def orthogonal(nucl):
	conv = '0,0,0,0'
	if nucl.upper() == 'A':  #0.1
		conv = '1,0,0,0'
	elif nucl.upper() == 'C': #0.9
		conv = '0,1,0,0'
	elif nucl.upper() == 'G': #0.8
		conv = '0,0,0,1'
	elif nucl.upper() == 'T': #0.2
		conv = '0,0,1,0'
	return conv

def enthalpy(nucl):
	conv = 0
	if nucl.upper() == 'CC':
		conv = 0.11
	elif nucl.upper() == 'TT':
		conv = 0.091
	elif nucl.upper() == 'AA':
		conv = 0.091
	elif nucl.upper() == 'GG':
		conv = 0.11
	elif nucl.upper() == 'CT':
		conv = 0.078
	elif nucl.upper() == 'TA':
		conv = 0.06
	elif nucl.upper() == 'AG':
		conv = 0.078
	elif nucl.upper() == 'CA':
		conv = 0.058
	elif nucl.upper() == 'TG':
		conv = 0.058
	elif nucl.upper() == 'CG':
		conv = 0.119
	elif nucl.upper() == 'TC':
		conv = 0.056
	elif nucl.upper() == 'AT':
		conv = 0.086
	elif nucl.upper() == 'GA':
		conv = 0.056
	elif nucl.upper() == 'AC':
		conv = 0.065
	elif nucl.upper() == 'GT':
		conv = 0.065
	elif nucl.upper() == 'GC':
		conv = 0.111
	return conv

def Galois4(nucl):
	conv = 0
	if nucl.upper() == 'CC':
		conv = 0.0
	elif nucl.upper() == 'TT':
		conv = 5.0
	elif nucl.upper() == 'AA':
		conv = 10.0
	elif nucl.upper() == 'GG':
		conv = 15.0
	elif nucl.upper() == 'CT':
		conv = 1.0
	elif nucl.upper() == 'TA':
		conv = 6.0
	elif nucl.upper() == 'AG':
		conv = 11.0
	elif nucl.upper() == 'CA':
		conv = 2.0
	elif nucl.upper() == 'TG':
		conv = 7.0
	elif nucl.upper() == 'CG':
		conv = 3.0
	elif nucl.upper() == 'TC':
		conv = 4.0
	elif nucl.upper() == 'AT':
		conv = 9.0
	elif nucl.upper() == 'GA':
		conv = 14.0
	elif nucl.upper() == 'AC':
		conv = 8.0
	elif nucl.upper() == 'GT':
		conv = 13.0
	elif nucl.upper() == 'GC':
		conv = 12.0
	return conv

"""
This function calculates physicochemical properties of DNA sequences based on the method proposed in the paper:
"Physicochemical property based computational scheme for classifying DNA sequence elements of Saccharomyces cerevisiae."
NOTE: this function uses a one-nucleotide slide windows to sum up all dinucleotides and then performes an average.
"""
def physicochemicalTotal(fileSeq):
	resultFile = open(fileSeq+'.pc', 'w')
	resultFile.write('Label,HBE,Stacking,Solvation\n')
	HBE = {'AA': -5.44, 'AC': -7.14, 'AG': -6.27, 'AT': -5.53, 'CA': -7.01, 'CC': -8.48, 'CG': -8.05, 'CT': -6.27, 'GA': -7.80, 'GC': -8.72, 'GG': -8.48, 'GT': -7.14, 'TA': -5.83, 'TC': -7.80, 'TG': -7.01, 'TT': -5.44}
	stacking = {'AA': -26.71, 'AC': -27.73, 'AG': -26.89, 'AT': -27.20, 'CA': -27.15, 'CC': -26.28, 'CG': -27.93, 'CT': -26.89, 'GA': -26.78, 'GC': -28.13, 'GG': -26.28, 'GT': -27.73, 'TA': -26.90, 'TC': -26.78, 'TG': -27.15, 'TT': -26.71}
	solvation = {'AA': -171.84, 'AC': -171.11, 'AG': -174.93, 'AT': -173.70, 'CA': -179.01, 'CC': -166.76, 'CG': -176.88, 'CT': -174.93, 'GA': -167.60, 'GC': -165.58, 'GG': -166.76, 'GT': -171.11, 'TA': -174.35, 'TC': -167.60, 'TG': -179.01, 'TT': -171.84}
	for te in SeqIO.parse(fileSeq, "fasta"):
		hbeResult = 0
		stackingResult = 0
		solvationResult = 0
		order = -1
		if str(te.id).upper().find("ALE-") != -1 or str(te.id).upper().find("RETROFIT-") != -1:
			order = 1
		# elif str(te.id).upper().find("ALESIA-") != -1:
		# 	order = 2
		elif str(te.id).upper().find("ANGELA-") != -1:
			order = 3
		elif str(te.id).upper().find("BIANCA-") != -1:
			order = 4
		# elif str(te.id).upper().find("BRYCO-") != -1:
		# 	order = 5
		# elif str(te.id).upper().find("LYCO-") != -1:
		# 	order = 6
		# elif str(te.id).upper().find("GYMCO-") != -1:
		# 	order = 7
		elif str(te.id).upper().find("IKEROS-") != -1:
			order = 8
		elif str(te.id).upper().find("IVANA-") != -1 or str(te.id).upper().find("ORYCO-") != -1:
			order = 9
		# elif str(te.id).upper().find("OSSER-") != -1:
		# 	order = 10
		# elif str(te.id).upper().find("TAR-") != -1:
		# 	order = 11
		elif str(te.id).upper().find("TORK-") != -1:
			order = 12
		elif str(te.id).upper().find("SIRE-") != -1:
			order = 13
		elif str(te.id).upper().find("CRM-") != -1:
			order = 14
		# elif str(te.id).upper().find("CHLAMYVIR-") != -1:
		# 	order = 15
		elif str(te.id).upper().find("GALADRIEL-") != -1:
			order = 16
		elif str(te.id).upper().find("REINA-") != -1:
			order = 17
		elif str(te.id).upper().find("TEKAY-") != -1 or str(te.id).upper().find("DEL-") != -1:
			order = 18
		elif str(te.id).upper().find("ATHILA-") != -1:
			order = 19
		elif str(te.id).upper().find("TAT-") != -1:
			order = 20
		elif str(te.id).upper().find("OGRE-") != -1:
			order = 21
		elif str(te.id).upper().find("RETAND-") != -1:
			order = 22
		# elif str(te.id).upper().find("PHYGY-") != -1:
		# 	order = 23
		# elif str(te.id).upper().find("SELGY-") != -1:
		# 	order = 24
		if order != -1:
			print(te.id)
			for i in range(len(te.seq)-1):
				dinucl = str(te.seq).upper()[i:i+2]
				if dinucl.find("N") == -1 and dinucl in HBE.keys():
					hbeResult += HBE[dinucl]
					stackingResult += stacking[dinucl]
					solvationResult += solvation[dinucl]
			hbeResult = hbeResult / (len(str(te.seq))-1)
			stackingResult = stackingResult / (len(str(te.seq))-1)
			solvationResult = solvationResult / (len(str(te.seq))-1)
			resultFile.write(str(order)+','+str(hbeResult)+','+str(stackingResult)+','+str(solvationResult)+'\n')
	resultFile.close()

"""
This function calculates physicochemical properties of DNA sequences based on the method proposed in the paper:
"Physicochemical property based computational scheme for classifying DNA sequence elements of Saccharomyces cerevisiae."
NOTE: this function uses a one-nucleotide slide windows to sum up all dinucleotides.
"""
def physicochemicalDinucl(fileSeq):
	resultFile = open(fileSeq+'.physicDN', 'w')
	resultFile.write('Class,HBE,Stacking,Solvation\n')
	HBE = {'AA': -5.44, 'AC': -7.14, 'AG': -6.27, 'AT': -5.53, 'CA': -7.01, 'CC': -8.48, 'CG': -8.05, 'CT': -6.27, 'GA': -7.80, 'GC': -8.72, 'GG': -8.48, 'GT': -7.14, 'TA': -5.83, 'TC': -7.80, 'TG': -7.01, 'TT': -5.44}
	stacking = {'AA': -26.71, 'AC': -27.73, 'AG': -26.89, 'AT': -27.20, 'CA': -27.15, 'CC': -26.28, 'CG': -27.93, 'CT': -26.89, 'GA': -26.78, 'GC': -28.13, 'GG': -26.28, 'GT': -27.73, 'TA': -26.90, 'TC': -26.78, 'TG': -27.15, 'TT': -26.71}
	solvation = {'AA': -171.84, 'AC': -171.11, 'AG': -174.93, 'AT': -173.70, 'CA': -179.01, 'CC': -166.76, 'CG': -176.88, 'CT': -174.93, 'GA': -167.60, 'GC': -165.58, 'GG': -166.76, 'GT': -171.11, 'TA': -174.35, 'TC': -167.60, 'TG': -179.01, 'TT': -171.84}
	for te in SeqIO.parse(fileSeq, "fasta"):
		hbeResult = 0
		stackingResult = 0
		solvationResult = 0
		order = -1
		if te.id.find("RLC_") != -1:
			order = 1
		elif te.id.find("RLG_") != -1:
			order = 2
		if order != -1:
			print(te.id)
			for i in range(len(te.seq)-1):
				dinucl = str(te.seq).upper()[i:i+2]
				if dinucl.find("N") == -1:
					hbeResult += HBE[dinucl]
					stackingResult += stacking[dinucl]
					solvationResult += solvation[dinucl]
			resultFile.write(str(order)+','+str(hbeResult)+','+str(stackingResult)+','+str(solvationResult)+'\n')
	resultFile.close()

"""
This function calculates the maximum length found in the dataset
"""
def maxLength(file):
	maxLen = 0
	for te in SeqIO.parse(file, "fasta"):
		if len(te.seq) > maxLen:
			maxLen = len(te.seq)
	return maxLen

"""
This function calculates k-mer frequencies in parallel mode.
"""
def kmerDB(file, id, seqs_per_procs, kmers, TEids, TEseqs, n, remain):
	if id < remain:
		init = id * (seqs_per_procs + 1)
		end = init + seqs_per_procs + 1
	else:
		init = id * seqs_per_procs + remain
		end = init + seqs_per_procs
	print("running in process "+str(id) + " init="+str(init)+" end="+str(end))
	resultFile = open(file+'.'+multiprocessing.current_process().name, 'w')

	while init < end and init < n:
		order = -1
		"""if str(TEids[init]).upper().find("RLC_") != -1 or str(TEids[init]).upper().find("COPIA") != -1:
			order = 1
		elif str(TEids[init]).upper().find("RLG_") != -1 or str(TEids[init]).upper().find("GYPSY") != -1:
			order = 2"""

		# Lineages from Copia
		"""if str(TEids[init]).upper().find("ALE-") != -1 or str(TEids[init]).upper().find("RETROFIT-") != -1 or str(TEids[init]).upper().find("ALESIA-") != -1 or str(TEids[init]).upper().find("ANGELA-") != -1 		or str(TEids[init]).upper().find("BIANCA-") != -1 or str(TEids[init]).upper().find("BRYCO-") != -1 or str(TEids[init]).upper().find("LYCO-") != -1 or str(TEids[init]).upper().find("GYMCO-") != -1 		or str(TEids[init]).upper().find("IKEROS-") != -1 or str(TEids[init]).upper().find("IVANA-") != -1 or str(TEids[init]).upper().find("ORYCO-") != -1 or str(TEids[init]).upper().find("OSSER-") != -1 or str(TEids[init]).upper().find("TAR-") != -1 or str(TEids[init]).upper().find("TORK-") != -1 or str(TEids[init]).upper().find("SIRE-") != -1:
			order = 1
		# Lineages from Gypsy
		elif str(TEids[init]).upper().find("CRM-") != -1 or str(TEids[init]).upper().find("CHLAMYVIR-") != -1 or str(TEids[init]).upper().find("GALADRIEL-") != -1 or str(TEids[init]).upper().find("REINA-") != -1 		or str(TEids[init]).upper().find("TEKAY-") != -1 or str(TEids[init]).upper().find("DEL-") != -1 or str(TEids[init]).upper().find("ATHILA-") != -1 or str(TEids[init]).upper().find("OGRE-") != -1 		or str(TEids[init]).upper().find("RETAND-") != -1 or str(TEids[init]).upper().find("PHYGY-") != -1 or str(TEids[init]).upper().find("SELGY-") != -1:
			order = 0"""
		if str(TEids[init]).upper().find("ALE-") != -1 or str(TEids[init]).upper().find("RETROFIT-") != -1:
			order = 1
		# elif str(TEids[init]).upper().find("ALESIA-") != -1:
		# 	order = 2
		elif str(TEids[init]).upper().find("ANGELA-") != -1:
			order = 3
		elif str(TEids[init]).upper().find("BIANCA-") != -1:
			order = 4
		# elif str(TEids[init]).upper().find("BRYCO-") != -1:
		# 	order = 5
		# elif str(TEids[init]).upper().find("LYCO-") != -1:
		# 	order = 6
		# elif str(TEids[init]).upper().find("GYMCO-") != -1:
		# 	order = 7
		elif str(TEids[init]).upper().find("IKEROS-") != -1:
			order = 8
		elif str(TEids[init]).upper().find("IVANA-") != -1 or str(TEids[init]).upper().find("ORYCO-") != -1:
			order = 9
		# elif str(TEids[init]).upper().find("OSSER-") != -1:
		# 	order = 10
		# elif str(TEids[init]).upper().find("TAR-") != -1:
		# 	order = 11
		elif str(TEids[init]).upper().find("TORK-") != -1:
			order = 12
		elif str(TEids[init]).upper().find("SIRE-") != -1:
			order = 13
		elif str(TEids[init]).upper().find("CRM-") != -1:
			order = 14
		# elif str(TEids[init]).upper().find("CHLAMYVIR-") != -1:
		# 	order = 15
		elif str(TEids[init]).upper().find("GALADRIEL-") != -1:
			order = 16
		elif str(TEids[init]).upper().find("REINA-") != -1:
			order = 17
		elif str(TEids[init]).upper().find("TEKAY-") != -1 or str(TEids[init]).upper().find("DEL-") != -1:
			order = 18
		elif str(TEids[init]).upper().find("ATHILA-") != -1:
			order = 19
		elif str(TEids[init]).upper().find("TAT-") != -1:
			order = 20
		elif str(TEids[init]).upper().find("OGRE-") != -1:
			order = 21
		elif str(TEids[init]).upper().find("RETAND-") != -1:
			order = 22
		# elif str(TEids[init]).upper().find("PHYGY-") != -1:
		# 	order = 23
		# elif str(TEids[init]).upper().find("SELGY-") != -1:
		# 	order = 24
		if order != -1:
			frequencies = [0 for x in range(len(kmers))]
			for i in range(len(TEseqs[init])):
				for l in range(1, 7):
					if i+l < len(TEseqs[init]):
						if TEseqs[init][i:i+l].upper().find("N") == -1 and TEseqs[init][i:i+l].upper() in kmers:
							index = kmers.index(TEseqs[init][i:i+l].upper())
							frequencies[index] += 1
			# print (TEids[init])
			frequenciesStrings = [str(x) for x in frequencies]
			resultFile.write(str(order)+','+','.join(frequenciesStrings)+'\n')
			# resultMap[TEids[init]] = str(order)+','+','.join(frequenciesStrings)
		init += 1
	resultFile.close()
	print("Process done in "+str(id))


if __name__ == '__main__':
	# 
	file = sys.argv[1]
	maxlen = maxLength(file)
	maxLength(file)
	print("Max length: "+str(maxlen))
	filterDBLineages(file, "DAX", maxlen)
	filterDBLineages(file, "EIIP", maxlen)
	filterDBLineages(file, "complementary", maxlen)
	#filterDBLineages(file, "orthogonal", maxlen)
	filterDBLineages(file, "enthalpy", maxlen)
	filterDBLineages(file, "Galois4", maxlen)
	physicochemicalTotal(file)
	
	# kmer features calculator in parallel mode
	# number of threads to calculate k-mer frequencies in parallel.
	threads = 32
	start_time = time.time()
	kmers = []
	for k in range(1,7):
		for item in itertools.product('ACGT', repeat=k):
			kmers.append(''.join(item))
	TEids = []
	TEseqs = []
	n = 0
	for te in SeqIO.parse(file, "fasta"):
		TEids.append(te.id)
		TEseqs.append(te.seq)
		n += 1
	seqs_per_procs = int(n/threads)+1
	remain = n % threads
	processes = [multiprocessing.Process(target=kmerDB, args=[file, x, seqs_per_procs, kmers, TEids, TEseqs, n, remain]) for x in range(threads)]
	[process.start() for process in processes]
	[process.join() for process in processes]

	finalFile = open(file+'.kmers', 'w')
	finalFile.write('Label,'+','.join(kmers)+'\n')
	for i in range(1, threads+1):
		filei = open(file+'.Process-'+str(i), 'r')
		lines = filei.readlines()
		for line in lines:
			finalFile.write(line)
		filei.close()
		os.remove(file+'.Process-'+str(i))
	finalFile.close()
	end_time = time.time()
	print("Threads time=", end_time - start_time)
