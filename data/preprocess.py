file = open('citationNumber-dblp')
outFile1 = open('dblp-paper-history.tsv','w')
outFile2 = open('dblp-paper-response.tsv','w')

count = 0 
for line in file:
	line = line.split('\t')
	# print(line)

	if len(line)>=12:
		lastCitationNumber = line[-2]
		history = line[:-11]
		# print(history, lastCitationNumber)

		outFile1.write(history[0])
		for h in history[1:]:
			outFile1.write('\t' + h)
		outFile1.write('\n')

		outFile2.write(lastCitationNumber+'\n')	

# print(count)