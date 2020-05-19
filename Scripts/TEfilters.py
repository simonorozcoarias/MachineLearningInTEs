import Bio
from Bio import SeqIO
import subprocess
import sys

"""
This function uses the all_tabfiles.tab produced by Inpactor's step1 to filters 
sequences with domains from both superfamilies (Gypsy and Copia).
Then it filters TEs with the same number of domains from different lineages.
Finally, using sequences from seqfike it removes TEs by lengths 
(lineages lengths were taken from GyDB).
NOTE: lineages that are comment in third filter are not find in angionsperms.
"""
def filters(tabfile, seqfile, threads, num_domains, length_tolerance):
	f = open(tabfile,"r")
	superfamilies = []
	noc = []
	lines = f.readlines()
	for line in lines:
		fields = line.split(";")
		domains = {}
		print(fields[0])
		for i in range(15, 26, 2):
			if len(fields[i].split(" ")) > 1:
				dom = fields[i].split(" ")[1]
				if dom != "NO":
					if dom in domains.keys():
						domains[dom] += 1
					else:
						domains[dom] = 1
		# first step, classification at superfamily level filtering elements with domains from both super-families
		count_gy = 0
		count_cp = 0
		if sum(domains.values()) >= num_domains:
			for dom in domains.keys():
				if dom.find("#RLC") != -1:
					count_cp += 1
				elif dom.find("#RLG") != -1:
					count_gy += 1
			if count_cp > 0:
				if count_gy == 0:
					superfamilies.append(line)
				else:
					noc.append(line)
			elif count_gy > 0:
				if count_cp == 0:
					superfamilies.append(line)
				else:
					noc.append(line)
		else:
			noc.append(line)
	print("Number of TEs deleted due to they had domains from both superfamilies: "+str(len(noc)))
	# second step, classification at lineage level filtering elements with same number of domains from different lineages
	linClassification = {}
	for line in superfamilies:
		fields = line.split(";")
		leng = 0
		maxi = 0
		no_fam = 0
		lineage = ""
		#print("Element: "+fields[0]+" has the following domains")
		domains = {}
		for i in range(15, 26, 2):
			if len(fields[i].split(" ")) > 1:
				dom = fields[i].split(" ")[1]
				if dom != "NO":
					if dom.find("#") != -1: 
						domn2 = dom.split("#")[1]
						if domn2.find("+") != -1:
							domn3 = str(domn2.split("+")[1]).upper()
							if domn3 in domains.keys():
								domains[domn3] += 1
							else:
								domains[domn3] = 1
		for dom in domains.keys():
			if domains[dom] >= maxi and domains[dom] > 0:
				if domains[dom] == maxi:
					no_fam = 1
				elif domains[dom] > maxi:
					no_fam = 0
				maxi = domains[dom]
				lineage = dom
			#print("domine: "+dom+" num: "+str(domains[dom]))
			leng += 1
		  
		if leng == 0 or no_fam == 1 or sum(domains.values()) < num_domains:
			noc.append(line)
		else:
			linClassification[fields[0]] = lineage
	print("Number of TEs deleted by first two filters: "+str(len(noc)))
	# third step, filtering with length
	resultFile = open(seqfile+".lineages", "w")
	#LTRrts = [te for te in SeqIO.parse(seqfile, "fasta") if te.id in linClassification.keys()]
	for teid in linClassification.keys():
		#print(teid)
		seq = [te.seq for te in SeqIO.parse(seqfile, "fasta") if te.id == teid]
		#print(len(seq))
		if len(seq) > 0:
			seq = str(seq[0])
			#print(len(seq))
			init = 0
			end = 0
			if linClassification[teid].upper() == "ALE" or linClassification[teid].upper() == "RETROFIT":
				init = 4700 - (4700*length_tolerance)
				end = 4900 + (4900*length_tolerance)
			# elif str(te.id).upper().find("ALESIA-") != -1:
			# 	order = 2
			elif linClassification[teid].upper() == "ANGELA":
				init = 5920 - (5920*length_tolerance)
				end = 12080 + (12080*length_tolerance)
			elif linClassification[teid].upper() == "BIANCA":
				init = 4136 - (4136*length_tolerance)
				end = 8326 + (8326*length_tolerance)
			# elif str(te.id).upper().find("BRYCO-") != -1:
			# 	order = 5
			# elif str(te.id).upper().find("LYCO-") != -1:
			# 	order = 6
			# elif str(te.id).upper().find("GYMCO-") != -1: 
			# 	order = 7
			elif linClassification[teid].upper() == "IKEROS":
				init = 3870 - (3870*length_tolerance)
				end = 8600 + (8600*length_tolerance)
			elif linClassification[teid].upper() =="IVANA" or linClassification[teid].upper() == "ORYCO":
				init = 4200 - (4200*length_tolerance)
				end = 4900 + (4900*length_tolerance)
			# elif str(te.id).upper().find("OSSER-") != -1:
			# 	order = 10
			# elif str(te.id).upper().find("TAR-") != -1:
			# 	order = 11
			elif linClassification[teid].upper() == "TORK":
				init = 4340 - (4340*length_tolerance)
				end = 9100 + (9100*length_tolerance)
			elif linClassification[teid].upper() == "SIRE":
				init = 9300 - (9300*length_tolerance)
				end = 9800 + (9800*length_tolerance)
			elif linClassification[teid].upper() == "CRM":
				init = 7000 - (7000*length_tolerance)
				end = 9000 + (9000*length_tolerance)
			# elif str(te.id).upper().find("CHLAMYVIR-") != -1:
			# 	order = 15
			elif linClassification[teid].upper() == "GALADRIEL":
				init = 6000 - (6000*length_tolerance)
				end = 7600 + (7600*length_tolerance)
			elif linClassification[teid].upper() == "REINA":
				init = 4700 - (4700*length_tolerance)
				end = 5000 + (5000*length_tolerance)
			elif linClassification[teid].upper() == "TEKAY" or linClassification[teid].upper() == "DEL":
				init = 8400 - (8400*length_tolerance)
				end = 19000 + (19000*length_tolerance)
			elif linClassification[teid].upper() == "ATHILA":
				init = 8500 - (8500*length_tolerance)
				end = 12000 + (12000*length_tolerance)
			elif linClassification[teid].upper() == "TAT" or linClassification[teid].upper() == "OGRE":
				init = 10000 - (10000*length_tolerance)
				end = 21000 + (21000*length_tolerance)
			# elif linClassification[teid].upper() == "RETAND":
			# 	order = 22
			# elif str(te.id).upper().find("PHYGY-") != -1:
			# 	order = 23
			# elif str(te.id).upper().find("SELGY-") != -1:
			# 	order = 24
			if len(seq) >= init and len(seq) <= end:
				resultFile.write(">"+linClassification[teid].upper()+"-"+teid+"\n"+seq+"\n")
			else:
				noc.append(teid)
	print("number of TEs deleted by the three filters: "+str(len(noc)))	

"""
This function filters TEs that contains nested insertions of TE class 2.
Class 2 sequences were taken from Repbase.
"""
def nestedtTEsFilter(seqfile, class2ReferencesTEs, steps, threads, similarity, aligLenBases, percLen):
	# fourth step, filtering nested inserted TEs
	if steps == 1:
		tabprocess = 'censor.ncbi '+class2ReferencesTEs+' -lib '+seqfile+'.lineages'+' -bprg blastn -map '+seqfile+'_nested_TEs.map -nofound -nomasked -aln '+seqfile+'_nested_TEs.aln -bprm \'-a '+threads+'\' '
		code = subprocess.call(tabprocess, shell=True)
	nestedTEs = open(seqfile+'_nested_TEs.map', 'r').readlines()
	nestedTesList = []
	for line in nestedTEs:
		line = line.replace(' ',';')
		columns = line.split(';')
		columns = [x for x in columns if x != '']
		if float(columns[7]) >= similarity and abs(float(columns[1])-float(columns[2])) >= aligLenBases:
			aligLen = abs(float(columns[1])-float(columns[2]))
			seq = [str(te.seq) for te in SeqIO.parse(class2ReferencesTEs, "fasta") if te.id == columns[0]]
			if len(seq) > 0:
				refLen = len(str(seq[0]))
				porcAlig = (aligLen*100) / refLen
				if porcAlig >= percLen:
					nestedTesList.append(columns[3])

	finalFile = open(seqfile+'_final', 'w')
	cont = 0
	for te in SeqIO.parse(seqfile+'.lineages', 'fasta'):
		if str(te.id) not in nestedTesList:
			finalFile.write(">"+str(te.id)+"\n"+str(te.seq)+"\n")
		else:
			cont += 1
	print(str(cont)+" TEs were deleted due to they contained nested insertions")
	finalFile.close()

"""
This function extract class 2 TEs from a list manually generated from repbase
"""
def extractSeqsRepbase(seqfile, repbaseIds):
	repbase = open(repbaseIds, 'r').readlines()
	finalFile = open(seqfile+'_class2', 'w')
	for te in SeqIO.parse(seqfile, 'fasta'):
		#print(str(te.id))
		teIs = len([x for x in repbase if str(te.id) == x.replace('\n', '')])
		print(teIs)
		if teIs > 0:
			finalFile.write(">"+str(te.id)+"\n"+str(te.seq)+"\n")
	finalFile.close()



if __name__ == "__main__": 
	tabfile = sys.argv[1] # Inpactor's step 1 tabular file
	seqfile = sys.argv[2] # file in fasta with sequences to be classified (with the same ids of tabfile)
	threads = sys.argv[3] # number of threads
	num_domains = 3 # minimum number of domains that a TE must have to be classified
	length_tolerance = 0.2 # percentage of length tolerance to filter TEs
	filters(tabfile, seqfile, threads, num_domains, length_tolerance)
	#extractSeqsRepbase(seqfile, 'repbase_ids_TEs_class2.txt') # this method extracts TEs class 2 from repbase, those sequences will be
	                                                           # used in nestedTEsFilter function
	similarity = 0.5 #minimum simlarity to filter nested TEs
	aligLen = 100 # minimum alignment length to filter nested TEs
	percLen = 10 # minimum percentage of TE length to be discarted to the filter
	nestedtTEsFilter(seqfile, 'repbase_20170127.fasta_class2', 1, threads, similarity, aligLen, percLen)
