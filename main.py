import NNcode

tempDict = {}
#Dictionary of all the specific attacksCIC for CICIDS Dataset
attacksCIC = {
  "BENIGN": 0,
  "FTP-Patator": 1,
  "SSH-Patator": 2,
  "DoS slowloris": 3,
  "DoS Slowhttptest": 4,
  "DoS Hulk": 5,
  "DoS GoldenEye": 6,
  "Heartbleed": 7,
  "Web Attack Brute Force": 8,
  "Web Attack XSS": 9,
  "Web Attack Sql Injection": 10,
  "Infiltration": 11,
  "Bot": 12,
  "PortScan": 13,
  "DDoS": 14,
}
#Rows to review CIC data and add to dictionary
CICrows = [78]
#Dictionary of all specific attacksCIC for 2nd dataset
mappingUNSW = {
  #attacks
  "Benign": 0,
  " Fuzzers": 1,
  "Reconnaissance": 2,
  "Shellcode": 3,
  "Analysis": 4,
  "Backdoor": 5,
  "DoS": 6,
  "Exploits": 7,
  "Generic": 8,
  "Worms": 9,
}
#Columns to review UNSW data and add to dictionary
UNSWcols = [0,1,2,3,11,45]

NNcode.NNanalysis(path=r"DataSets\CIC-IDS-2017", header=0, indexCol=None, 
                  mapped=attacksCIC, colL=CICrows, Xmax=77, labelCol=78, attackNum=15)
              
NNcode.NNanalysis(path=r"DataSets\UNSW-NB15", header=None, indexCol=None,
                    mapped=mappingUNSW,colL=UNSWcols, Xmax=44, labelCol=45, attackNum=10,
                    dropFeats=[1,3], missReplacement=["Benign"],missCols=[47])