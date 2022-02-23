import csv
import os


def main():
    current_dir = os.getcwd()
    #sample_dir = current_dir + '/Sample prod' #path where raw samples are
    sample_dir = current_dir + '/sampleGrande'
    data_dir = current_dir + '/data'
    for f in os.listdir(data_dir):
        os.remove(os.path.join(data_dir, f))
    count =0;
    if(os.path.exists(sample_dir)):
        for dirpath, dirname, filename in os.walk(sample_dir, topdown=False):
            with open(data_dir + '/labels.csv', "w") as l:
                for name in filename:
                    
                    with open(sample_dir + '/' + name, "r") as f:
                        csv_file = csv.reader(f)
                        sample = []
                        try:
                            print(name)
                            header = csv_file.__next__()[0].split(';')
                        except:
                            continue
                        
                            
                            
                        
                        for row_raw in csv_file:
                            row = row_raw[0].split(';')
                            
                            if (row[0] == 'V' and row[2] == '4' ): #procura pelo grafico de vibração para a 4 mudança, aceleração
                                for x in row[8:] : # vai so escrever no ficheiro apenas os valores, retira labels
                                    sample.append(float(x))
                            elif(row[0] == 'F' and row[2] == '4' and (row[4] == '105' or row[4] == '106' or row[4] == '107')):
                            # sample.append(",".join(str(x) for x in row))
                                sample.append(float(row[5]))
                                
                        
                    
                    
                    
                    if(len(sample) == 1030):
                        
                        
                        if(header[8] == '2'):
                            continue
                        else:
                            #escrever no labels.csv
                            head = [name.split('.ESS')[0] + '.csv', header[8]] #original file
                            writer = csv.writer(l)
                            writer.writerow(head)
                            
                            with open(data_dir + '/' + name.split('.ESS')[0] + '.csv', 'w') as f:
                                writer = csv.writer(f)
                                writer.writerow(sample)
                        count += 1
                        #print(str(count) + "\n")
                        #contador de ficheiros
                        if(count >= 138):
                            print("fim")
                            return 1;

                        
                        

    
    
main()
