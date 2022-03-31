import csv
import os

def normalize(value,min,max):
    
    x = (value-min)/(max-min)
    # if(x > 1):
    #     x = 1
    return x

def main():
    current_dir = os.getcwd()
    #sample_dir = current_dir + '/Sample prod' #path where raw samples are
    sample_dir = current_dir + '/sampleGrande'
    data_dir = current_dir + '/tmp'
    for f in os.listdir(data_dir):
        os.remove(os.path.join(data_dir, f))
    count =0
    bad = 0
    skip = False
    if(os.path.exists(sample_dir)):
        for dirpath, dirname, filename in os.walk(sample_dir, topdown=False):
            with open(data_dir + '/labels.csv', "w") as l:
                for name in filename:
                    
                    with open(sample_dir + '/' + name, "r") as f:
                        csv_file = csv.reader(f)
                        sample = []
                        passagem = {}
                        oldgear=''
                        newgear=''
                        try:
                            #print(name)
                            header = csv_file.__next__()[0].split(';')
                        except:
                            continue
                        
                        if(header[8] == '2' or header[8] == '1'):
                            continue
                            
                        skip = False
                        for row_raw in csv_file:
                            row = row_raw[0].split(';')
                            if(row[0] == 'F0' and row[4] == '110'):
                                value =  float(row[6])
                                if value < 0:
                                    skip = True
                                    break
                            if(row[0] == 'F'):
                                if(row[4] == '105'):
                                    if(row[2] == '6'):
                                        if(row[3]=='1'):
                                            sample.append(normalize(float(row[5]),0,8))
                                        else:
                                            sample.append(normalize(float(row[5]),0,8.4))
                                    
                                    if(row[2] == '5'):
                                        if(row[3]=='1'):
                                            sample.append(normalize(float(row[5]),0,8.7))
                                        else:
                                            sample.append(normalize(float(row[5]),0,10.1))
                                    
                                    if(row[2] == '4' ):
                                        if(row[3]=='1'):
                                            sample.append(normalize(float(row[5]),0,11.7))
                                        else:
                                            sample.append(normalize(float(row[5]),0,10.9))
                                    if(row[2] == '3' ):
                                            sample.append(normalize(float(row[5]),0,6))
                                    if(row[2] == '2' or row[2] == '7'):
                                            sample.append(normalize(float(row[5]),0,10))
                                    if(row[2] == '1' ):
                                            sample.append(normalize(float(row[5]),0,25))

                                if(row[4] == '106'):
                                        if(row[2] == '6'):
                                            if(row[3]=='1'):
                                                sample.append(normalize(float(row[5]),0,3.1))

                                            else:
                                                sample.append(normalize(float(row[5]),0,2.4))
                                        
                                        
                                        
                                        if(row[2] == '5' ):
                                            if(row[3]=='1'):
                                                sample.append(normalize(float(row[5]),0,1.9))
                                            else:
                                                sample.append(normalize(float(row[5]),0,1.8))
                                        
                                        if(row[2] == '4' ):
                                            if(row[3]=='1'):
                                                sample.append(normalize(float(row[5]),0,2.1))
                                            else:
                                                sample.append(normalize(float(row[5]),0,2))
                                        if(row[2] == '3' ):
                                            if(row[3]=='1'):
                                                sample.append(float(row[5]))
                                            else:
                                                sample.append(normalize(float(row[5]),0,1.4))
                                        
                                        if(row[2] == '2' ):
                                            if(row[3]=='1'):
                                              sample.append(normalize(float(row[5]),0,2))
                                            else:
                                                sample.append(normalize(float(row[5]),0,2.2))
                                        if(row[2] == '1' ):
                                            if(row[3]=='1'): 
                                              sample.append(normalize(float(row[5]),0,1.2))
                                            else:
                                                sample.append(float(row[5]))
                                        if(row[2] == '7'):
                                             sample.append(normalize(float(row[5]),0,0.7))
                                if(row[4] == '107'):
                                        if(row[2] == '6' ):
                                            if(row[3]=='1'):
                                                sample.append(normalize(float(row[5]),0,2.2))
                                            else:
                                                sample.append(normalize(float(row[5]),0,1.2))
                                        if(row[2] == '5'):
                                            sample.append(normalize(float(row[5]),0,0.5))
                                        
                                        if(row[2] == '4' ):
                                            if(row[3]=='1'):
                                                sample.append(normalize(float(row[5]),0,2))
                                            else:
                                                sample.append(float(row[5]))
                                        if(row[2] == '3' ):
                                            if(row[3]=='1'):
                                                sample.append(normalize(float(row[5]),0,0.3))
                                            else:
                                                sample.append(normalize(float(row[5]),0,0.5))
                                        
                                        if(row[2] == '2' ):
                                              sample.append(normalize(float(row[5]),0,4))
                                       
                                        if(row[2] == '1' ):
                                            if(row[3]=='1'):
                                              sample.append(normalize(float(row[5]),0,0.1))
                                            else:
                                                sample.append(float(row[5]))
                                        if(row[2] == '7'):
                                            sample.append(normalize(float(row[5]),0,0.1))    
                            elif(row[0] == 'FO'):
                                    
                                min = -50
                                max = 50
                                # if(float(row[6]) > max):
                                #     print(str(float(row[6])) + " > " + max)
                                
                                
                                sample.append(normalize(float(row[6]),min,max))
                            elif(row[0] == 'M'):
                                newgear = row[2] + row[3]
                                if(newgear != oldgear):
                                    if(passagem):
                                        #print("dicionario= " + str(passagem.values()))
                                        
                                        #transferir valores para sample
                                        for value in passagem.values():
                                            sample.append(value)
                                    passagem = {}
                                    
                                if(row[4] == '30'):
                                    passagem[row[4]] = normalize(float(row[5]),0,25)
                                if(row[4] == '26'):
                                    passagem[row[4]] = normalize(float(row[5]),-10000,2800)
                                if(row[4] == '4'):
                                    passagem[row[4]] = normalize(float(row[5]),0,40)
                                if(row[4] == '3'):
                                    passagem[row[4]] = normalize(float(row[5]),-10,500)    
                                if(row[4] == '0'):
                                    passagem[row[4]] = normalize(float(row[5]),0,2)
                                
                                oldgear = newgear    
                        
                        #ultimos valores em passagem
                        for value in passagem.values():
                            sample.append(value) 
                            
                            
                            
                                
                        
                    
                    
                    
                    if(len(sample) == 102 and not skip):
                        
                        if(header[8] == '0'):
                            bad+=1
                       
                        
                        #escrever no labels.csv
                        head = [name.split('.ESS')[0] + '.csv', header[8]] #original file
                        writer = csv.writer(l)
                        writer.writerow(head)
                        
                        with open(data_dir + '/' + name.split('.ESS')[0] + '.csv', 'w') as f:
                            writer = csv.writer(f)
                            writer.writerow(sample)
                        count += 1
                        print("Total ensaios = " + str(count))
                        print("Total más = " + str(bad))
                        #contador de ficheiros
                        # if(count >= 138):
                        #     print("fim")
                        #     return 1;

                        
                        

    
    
main()
