import NNcode

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

#Dictionary of all specific attacksCIC for 2nd dataset
attacksCIC2nd = {

}

NNcode.NNAnalysis("DataSets\CIC-IDS-2017", attacksCIC)