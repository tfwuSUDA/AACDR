import os
import re
import json
import torch
import math
import pickle
import rdkit
from rdkit import Chem
import pandas as pd
import numpy as np
# from scipy.stats import stats
# from scipy.stats import  mannwhitneyu
# from scipy.sparse import csr_matrix
from pathlib import Path
import scipy.sparse as sp
from collections import defaultdict
from dataset import *

cwd = os.getcwd().split(r'/')[:-1]
cwd = '/'.join(cwd)
def normalADJ(adj):
    for i in range(100):
        adj[i][i] = 1
    D_sqrt_inv = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    adj = D_sqrt_inv.dot(adj).dot(D_sqrt_inv)
    return adj
def read_drug_list(type=None):
    if type == 'GDSC':
        GDSC = []
        with open(cwd + '/data/GDSC/GDSC_drug_binary.csv', 'r')as f:
            lines = f.readlines()[1:]
            for line in lines:
                l = line.strip().split(',')
                GDSC.append(l[-1])
        return GDSC
    else:
        TCGA = []
        with open(cwd + '/data/TCGA/TCGA_drug_new.csv', 'r')as f:
            lines = f.readlines()[1:]
            for line in lines:
                l = line.strip().split(',')
                TCGA.append(l[-1])
        return TCGA       

def read_drug_graph(type, drug_list):
    result = {}
    floader = cwd + '/data/'+type+'/drug_graph/'
    for id in drug_list:
        path = floader + str(id) + '.npz'
        result[str(id)] = np.load(path)
    return result

def read_gene_expr():
    GDSC = pd.read_csv(cwd + '/data/GDSC/GDSC_expr_z_702.csv', sep=',', header=0, index_col=[0])
    disc_TCGA = pd.read_csv(cwd + '/data/TCGA/TCGA_unlabel_expr_702_01A.csv', sep=',', header=0, index_col=[0])
    TCGA = pd.read_csv(cwd + '/data/TCGA/TCGA_expr_z_702.csv', sep=',', header=0, index_col=[0])
    return GDSC, disc_TCGA, TCGA



def makeDataset():
    GDSC_expr, TCGA_unlabel_expr, TCGA_expr = read_gene_expr()
    
    GDSC_drug_list = read_drug_list('GDSC')
    TCGA_drug_list = read_drug_list('TCGA')

    response = pd.read_csv(cwd + '/data/GDSC/GDSC_binary_response_151.csv', sep=',', header=0, index_col=[0])
    response.columns = response.columns.astype(str)
    GDSC_drug_list = [id for id in response.columns if id in GDSC_drug_list]
    response = response[GDSC_drug_list]
    
    gid_list = list(GDSC_expr.columns)

    GDSC_data = []
    for drug_id in GDSC_drug_list:
        for cellline_name in response.index:
            x = response.loc[cellline_name, drug_id]
            if cellline_name in GDSC_expr.index and (not np.isnan(x)):
                df = GDSC_expr.loc[cellline_name][gid_list]
                GDSC_data.append([int(drug_id), torch.tensor(df).float(), x])
    
    gdsc_only_dataset = LabeledDataset(GDSC_data)
    with open(cwd + '/data/GDSC/GDSC_only_dataset.pkl','wb')as w2:
        pickle.dump(gdsc_only_dataset, w2)

    TCGA_unlabel_data = []
    dis = TCGA_unlabel_expr.T
    for person in dis.columns:
        df = dis[person][gid_list]
        TCGA_unlabel_data.append(df)
    TCGA_unlabel_data = np.array(TCGA_unlabel_data)
    
    tcga_unlabel_dataset = UnlabeledDataset(torch.tensor(TCGA_unlabel_data).float())
    with open(cwd + '/data/TCGA/TCGA_unlabel_dataset.pkl','wb')as w3:
        pickle.dump(tcga_unlabel_dataset, w3)

    response = pd.read_csv(cwd + '/data/TCGA/TCGA_response_new.csv', sep=',', header=0, index_col=[0])
    response.columns = response.columns.astype(str)
    TCGA_drug_list = [id for id in response.columns if id in TCGA_drug_list]
    response = response[TCGA_drug_list]

    TCGA_data = []
    for drug_id in TCGA_drug_list:
        for cellline_name in response.index:
            x = response.loc[cellline_name, drug_id]
            if cellline_name in TCGA_expr.index and (not np.isnan(x)):
                df = TCGA_expr.loc[cellline_name][gid_list]
                TCGA_data.append([int(drug_id), torch.tensor(df).float(), x])
    

    testDataset = LabeledDataset(TCGA_data)

    with open(cwd + '/data/TCGA/TCGA_dataset.pkl','wb')as w:
        pickle.dump(testDataset, w)
    
def makeCNVDataset():
    driver_genes2 = ['IDH1', 'PSIP1', 'SDC4', 'SOX17', 'KDR', 'CIITA', 'STAG1', 'MSH2', 'MAP2K7', 'BCLAF1', 'ARHGAP35', 'LMO1', 'MARK2', 'SRSF2', 'ARID1A', 'FOXA1', 'CDX2', 'AKT2', 'MYOD1', 'NCKIPSD', 'CBLC', 'MNX1', 'OLIG2', 'RBM38', 'CCND2', 'SMARCE1', 'DAZAP1', 'SNX29', 'DGCR8', 'ATP1A1', 'RB1', 'HNF1A', 'NFATC2', 'NUP98', 'TFG', 'TCF4', 'CREB3L1', 'GPHN', 'SFMBT2', 'GAS7', 'KRAS', 'NCOA2', 'CRNKL1', 'ENPEP', 'PABPC1', 'RARA', 'MYC', 'SDHB', 'HSPG2', 'KIT', 'SND1', 'MLLT6', 'CD209', 'APC', 'KEL', 'MSI2', 'KEAP1', 'RASA2', 'RAD21', 'PMS1', 'CBFB', 'TLX3', 'ERG', 'BCL11A', 'NACA', 'CD74', 'RABEP1', 'RRAGC', 'BRCA1', 'SALL4', 'ELK4', 'TET1', 'KAT6A', 'ZNF384', 'HOXA9', 'KLF5', 'MUTYH', 'TFPT', 'BCL11B', 'AJUBA', 'PPARG', 'MUC16', 'FGFR4', 'PRKACA', 'GRIN2A', 'TFAP4', 'PPP2R1A', 'NTRK3', 'PLAG1', 'RAF1', 'NPRL2', 'PTPRD', 'MLLT1', 'CYSLTR2', 'ECT2L', 'BUB1B', 'EWSR1', 'SMAD3', 'CHEK2', 'ATIC', 'CCNB1IP1', 'EXT1', 'CTNNB1', 'PIK3R1', 'PTPN13', 'RET', 'FANCD2', 'RPN1', 'TP53', 'RNF213', 'GNA11', 'SP140', 'FANCA', 'TAL1', 'ITK', 'CSMD3', 'ATF7IP', 'ACKR3', 'KIF5B', 'TMEM127', 'SKI', 'ZCCHC8', 'LEF1', 'CDK12', 'PTPRB', 'PRKD2', 'CPEB3', 'FIP1L1', 'EPHA3', 'BIRC6', 'NCOA4', 'MYCN', 'MAF', 'NPM1', 'JAK1', 'GLI1', 'CLTC', 'RBM39', 'SFPQ', 'TCEA1', 'PRR14', 'COL1A1', 'PLCB4', 'NFE2L2', 'GNAS', 'NDRG1', 'BRIP1', 'ROBO2', 'BCL2L12', 'POLE', 'FUS', 'AKT3', 'UBR5', 'USP44', 'SF3B1', 'NRG1', 'MYB', 'PCDH17', 'NOTCH1', 'CLIP1', 'TSC2', 'LPP', 'KLHL6', 'GNA13', 'MCM3AP', 'PPT2', 'ALK', 'ZCRB1', 'REL', 'CASZ1', 'ETV6', 'SET', 'NIN', 'CSF1R', 'COL3A1', 'HMGA2', 'ARHGEF10', 'CEP89', 'HOXC13', 'ERCC4', 'KMT2A', 'FAT4', 'TSHR', 'TFEB', 'SETD1B', 'SLC45A3', 'CLP1', 'SETDB1', 'NT5C2', 'PTEN', 'JAK3', 'RRAS2', 'RSPO2', 'BMPR2', 'TGIF1', 'KLF6', 'DICER1', 'ERCC2', 'SRSF3', 'ZNF479', 'PRDM1', 'GATA2', 'TCF7L2', 'TRIM33', 'TOP1', 'WNK2', 'PDGFB', 'EIF4A2', 'KDM5A', 'EIF3E', 'CREB3L2', 'HERPUD1', 'ZNF521', 'CANT1', 'RECQL4', 'RELA', 'NRAS', 'CALR', 'KLF4', 'HRAS', 'ROS1', 'HOXD11', 'PAX5', 'MUC1', 'HGF', 'PDCD1LG2', 'STIL', 'DDX5', 'SMARCD1', 'ZNF721', 'SH3GL1', 'ACSL6', 'BCL2', 'HLF', 'RMI2', 'EFTUD2', 'ZFHX3', 'SMARCA4', 'LRIG3', 'FBLN1', 'SOX21', 'KIAA1549', 'PML', 'MAP2K2', 'ERBB4', 'FOXO3', 'PTCH1', 'ARHGAP5', 'IRF4', 'CCNE1', 'FOXL2', 'HNRNPA2B1', 'SH2B3', 'NTHL1', 'PIK3CB', 'POLG', 'CASP3', 'CEBPA', 'TRAF3', 'HOXA13', 'PDGFRA', 'FOXR1', 'SOX9', 'ZNF429', 'FAT1', 'BCL6', 'FGFR2', 'PTMA', 'MDM4', 'RHOA', 'TLL1', 'NCOA1', 'PTPN6', 'MB21D2', 'VHL', 'KAT6B', 'CTNND2', 'CDH1', 'CDH17', 'LTB', 'TBL1XR1', 'FEN1', 'RALGDS', 'ETV4', 'MACC1', 'HIP1', 'CARD11', 'TCIRG1', 'PARP4', 'PRF1', 'SFRP4', 'CBLB', 'SUSD2', 'DHX9', 'CREB1', 'DDX6', 'MGMT', 'TMPRSS2', 'RANBP2', 'POU2AF1', 'CBFA2T3', 'MAFB', 'POLQ', 'FBN2', 'PEG3', 'FHIT', 'MALT1', 'POU5F1', 'ZNF626', 'PRDM16', 'MAP3K13', 'MECOM', 'MYH11', 'FAT2', 'NFKB2', 'CDC73', 'DNMT3A', 'AXIN2', 'FANCC', 'FBLN2', 'SRC', 'TRRAP', 'QKI', 'LEPROTL1', 'RUNX1', 'ARHGAP26', 'WIF1', 'RAP1GDS1', 'COL2A1', 'NF1', 'FGFR1', 'EED', 'FBXW7', 'ZFP36L1', 'FNBP1', 'CIC', 'SMAD4', 'SPEN', 'ELN', 'ESRRA', 'TET2', 'POU2F2', 'ABL1', 'TPM4', 'ASPSCR1', 'NBEA', 'ETV1', 'RPL5', 'CDKN2A', 'FN1', 'ING1', 'RASA1', 'EZR', 'KTN1', 'ETNK1', 'TRAF7', 'AXIN1', 'CNBP', 'ARHGEF10L', 'RSPO3', 'EP300', 'MLH1', 'PBRM1', 'BRD7', 'PTPRT', 'SDHAF2', 'CD58', 'PICALM', 'GMPS', 'THRAP3', 'DROSHA', 'KLK2', 'NCOR2', 'TLX1', 'CHCHD7', 'NFIB', 'TEC', 'BMPR1A', 'SBDS', 'EXT2', 'TCF3', 'ERC1', 'RBM15', 'FSTL3', 'ELL', 'BRD4', 'MAP2', 'MLLT11', 'BRAF', 'ABL2', 'DAXX', 'NXF1', 'FEV', 'RAD17', 'BCL3', 'PCBP1', 'CHD4', 'BCR', 'CSF3R', 'ABCB1', 'NT5C3A', 'CCND1', 'LARP4B', 'MEN1', 'IFNAR1', 'CCND3', 'ZNF93', 'SLC34A2', 'PER1', 'CCR7', 'RPS3A', 'RHPN2', 'PRRX1', 'ASXL1', 'TCL1A', 'DCTN1', 'COX6C', 'EPHA7', 'MYCL', 'PTK6', 'AKAP9', 'HOXA11', 'PMS2', 'SMARCB1', 'ZEB1', 'RNF6', 'TOP2A', 'CDK6', 'KMT2C', 'WNK4', 'CDKN1A', 'ELF3', 'BRCA2', 'CACNA1D', 'FAM174B', 'CREBBP', 'ID3', 'CUX1', 'GRM3', 'PAX3', 'BLM', 'BARD1', 'KNSTRN', 'POT1', 'PDGFRB', 'CD28', 'CDH11', 'FCRL4', 'HSP90AB1', 'SIX2', 'CYLD', 'CDKN1B', 'ZNRF3', 'EHD2', 'BAZ1A', 'PTPN14', 'RBFOX1', 'TBX3', 'N4BP2', 'NFKBIE', 'ACVR2A', 'PRKCB', 'FAM186A', 'ATM', 'PREX2', 'XPO1', 'PAX7', 'ARID2', 'ARNT', 'SDHC', 'GPC5', 'EPAS1', 'LZTR1', 'GNAI2', 'BCL10', 'FOXA2', 'B2M', 'RUNX1T1', 'JAZF1', 'PATZ1', 'KLHL36', 'CAMTA1', 'IFNGR1', 'LIFR', 'BTG2', 'GTF2I', 'EZH2', 'SGK1', 'FAS', 'FLI1', 'DOT1L', 'UGT2B17', 'RAC1', 'USP8', 'RGL3', 'RHOH', 'ZBTB16', 'DIS3', 'TERT', 'KDSR', 'BRD3', 'AFF3', 'TPR', 'DTX1', 'HSP90AA1', 'U2AF2', 'MYH9', 'LCP1', 'IDH2', 'TRIP11', 'LCK', 'LMNA', 'AKT1', 'FOXO1', 'TNFRSF14', 'ERCC3', 'ARHGEF12', 'SUFU', 'PIM1', 'CBL', 'PRCC', 'TFRC', 'CDKN2C', 'CNOT3', 'PLCG1', 'CNBD1', 'USP6', 'ESR1', 'EBF1', 'AFF1', 'MITF', 'CCR4', 'WRN', 'LATS2', 'MAP2K1', 'MET', 'SIN3A', 'CD79A', 'DDB2', 'ZNF148', 'OMD', 'ANK1', 'ZBTB20', 'CRTC1', 'CXCR4', 'PPM1D', 'SETBP1', 'ZNF165', 'CASP9', 'MAX', 'FGFR3', 'ZNF331', 'STAT3', 'SOHLH2', 'MN1', 'MPL', 'PPP6C', 'LSM14A', 'PRKAR1A', 'FES', 'SRGAP3', 'KMT2D', 'KIFC1', 'NEFH', 'CCDC6', 'CDK4', 'TGFBR2', 'CYP2C8', 'IL7R', 'ABI1', 'HEY1', 'NIPBL', 'TNFRSF17', 'MLLT10', 'VAV1', 'IRF1', 'SYK', 'FAT3', 'FH', 'FANCF', 'ALB', 'DDR2', 'LMO2', 'ERBB3', 'ZBTB7B', 'STK11', 'NFKBIA', 'FLT4', 'PRKCD', 'DCSTAMP', 'PALB2', 'MAPK1', 'BAX', 'NTRK1', 'ASXL2', 'SPOP', 'BCL9', 'GOLGA5', 'IGF2BP2', 'LOX', 'DUSP16', 'U2AF1', 'CASP8', 'MDM2', 'EPHA2', 'LASP1', 'EML4', 'STAT6', 'XPA', 'PAX8', 'CRTC3', 'NF2', 'PRDM2', 'MAP2K4', 'YWHAE', 'SOS1', 'S100A7', 'RNF43', 'SOX2', 'HOXC11', 'CHST11', 'NUMA1', 'NKTR', 'SMAD2', 'SS18', 'AFF4', 'BMP5', 'PRPF40B', 'FGD5', 'HERC2', 'BCL9L', 'BAP1', 'LATS1', 'RIPK1', 'ERBB2', 'PBX1', 'FLCN', 'SDHD', 'SOCS1', 'DDX10', 'MGA', 'PTPRK', 'KDM3B', 'NCOR1', 'POLD1', 'CTNNA2', 'EPS15', 'SS18L1', 'CNTNAP2', 'HTRA2', 'EGFR', 'GATA3', 'IKZF1', 'RAD51B', 'ERCC5', 'MAP3K1', 'PPFIBP1', 'PTPN11', 'TSC1', 'TPM3', 'SATB1', 'ITGAV', 'RBFOX2', 'ZMYM2', 'NR4A3', 'KCNJ5', 'ATR', 'CLTCL1', 'TCF12', 'FAM131B', 'HIF1A', 'APOBEC3B', 'IKBKB', 'TRIM24', 'INO80', 'DNM2', 'FAM135B', 'TP63', 'ZNF208', 'FOXP1', 'FANCE', 'PHOX2B', 'GOPC', 'HMGA1', 'PWWP2A', 'MUC4', 'CHD2', 'CTCF', 'ZNF814', 'ETV5', 'ARID1B', 'SETD2', 'TNFAIP3', 'MTOR', 'TRIM27', 'MYD88', 'ALDH2', 'DEK', 'PAK2', 'SMO', 'MLF1', 'RXRA', 'STRN', 'ACSL3', 'WWTR1', 'NUTM1', 'LYN', 'BIRC3', 'CHIC2', 'RGS7', 'CNTRL', 'FUBP1', 'ATG7', 'JUN', 'FBXO11', 'MLLT3', 'VTI1A', 'PCM1', 'CDH10', 'WT1', 'CR1', 'NOTCH2', 'JAK2', 'KAT7', 'FLT3', 'CD79B', 'ISX', 'IL21R', 'TAL2', 'NRP1', 'BTG1', 'SIX1', 'HOXD13', 'GNAQ', 'PCMTD1', 'RFWD3', 'NUP214', 'FANCG', 'ACVR1', 'ATF1', 'SPECC1', 'XPC', 'EGR2', 'ZNF680', 'NAB2', 'CD274', 'LRP1B', 'SDHA', 'BCL7A', 'LDB1', 'TNC', 'SUZ12', 'SIRPA', 'MEF2B', 'PTPRC', 'CCNC', 'UBE2D2', 'HOOK3', 'TAF15', 'PAFAH1B2', 'IL2', 'NSD1', 'NBN', 'LYL1', 'MAML2', 'DCC', 'MSH6', 'MYO5A', 'A1CF', 'STAT5B', 'ZNF780A', 'DDIT3', 'IKZF3', 'PPP3CA', 'PIK3CA', 'IL6ST', 'CUL3', 'DNAJB1', 'RPL22']
    print(len(driver_genes2))
    
    gdsc_response = pd.read_csv(cwd + '/data/GDSC/GDSC_binary_response_151.csv', sep=',', header=0, index_col=[0])
    gdsc_response.columns = gdsc_response.columns.astype(str)
    gdsc_cell_names = set(gdsc_response.index)
    
    model_map = pd.read_csv(cwd + '/data/CNV/Model.csv')
    modelID2Name = {}
    for index, row in model_map.iterrows():
        modelid = row['ModelID']
        name1 = row['CellLineName']
        name2 = row['StrippedCellLineName']
        if name1 in gdsc_cell_names:
            modelID2Name[modelid] = name1
        else:
            modelID2Name[modelid] = name2
        
    cell_cn = pd.read_csv(cwd + '/data/CNV/OmicsCNGene.csv', index_col=[0]) # https://depmap.org/portal/data_page/?tab=allData
    def rename_cell_cn_index(s):
        return modelID2Name[s]
    def rename_cell_cn_columns(s):
        return s.split(' ')[0]
    cell_cn = cell_cn.rename(columns=rename_cell_cn_columns) # gene
    cell_cn = cell_cn.rename(index=rename_cell_cn_index) # cell lines
    
    
    nan_gene_columns = list(cell_cn.columns[cell_cn.isnull().any()])
    cell_cn = cell_cn.drop(columns=nan_gene_columns)
    cell_cn_log2 = np.log2(cell_cn)
    cell_cn = pd.DataFrame(cell_cn_log2.values, index=cell_cn.index, columns=cell_cn.columns).T
    
    cell_cn = cell_cn.apply(lambda x:(x-x.mean())/x.std(), axis=0)
    cell_cn = cell_cn[cell_cn.index.isin(driver_genes2)]
    cell_cn = cell_cn.reindex(index=driver_genes2)
    
    cell_cn.to_csv(cwd + '/data/CNV/deepmap.csv')
    
    # patients
    test_patients = pd.read_csv(cwd + '/data/TCGA/TCGA_expr_z_702.csv', sep=',', header=0, index_col=[0])
    test_set_patients = set(test_patients.index)

    tcga = pd.read_csv(cwd + '/data/CNV/broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.gene.xena', sep='\t', index_col=[0]) # https://xenabrowser.net/datapages/?dataset=broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.gene.xena&host=https%3A%2F%2Fpancanatlas.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
    # genes_for_tcga = genes
    select_cols = [col for col in tcga.columns if col.endswith('-01')]
    tcga =  tcga[select_cols]
    def renamef(s):
        s = s[:-3]
        return s
    tcga = tcga.rename(columns=renamef)
    
    
    tcga = tcga.apply(lambda x:(x-x.mean())/x.std(), axis=0)
    tcga = tcga[tcga.index.isin(driver_genes2)]
    tcga = tcga.reindex(index=driver_genes2)
    
    tcga.to_csv(cwd + '/data/CNV/tcga.csv')
    
    # print()

    GDSC_drug_list = read_drug_list('GDSC')
    TCGA_drug_list = read_drug_list('TCGA')

    tcga = pd.read_csv(cwd + '/data/CNV/tcga.csv', index_col=[0]).T
    
    wes = pd.read_csv(cwd + '/data/CNV/deepmap.csv', index_col=[0]).T


    GDSC_drug_list = [id for id in gdsc_response.columns if id in GDSC_drug_list]
    gdsc_response = gdsc_response[GDSC_drug_list]

    tcga_response = pd.read_csv(cwd + '/data/TCGA/TCGA_response_new.csv', sep=',', header=0, index_col=[0])
    tcga_response.columns = tcga_response.columns.astype(str)
    TCGA_drug_list = [id for id in tcga_response.columns if id in TCGA_drug_list]
    tcga_response = tcga_response[TCGA_drug_list]

    gdsc_data = []
    gdsc_cell = set()
    gdsc_drug = set()
    p = 0
    n = 0
    for drug_id in GDSC_drug_list:
        for cellline_name in gdsc_response.index:
            x = gdsc_response.loc[cellline_name, drug_id]
            if cellline_name in wes.index and (not np.isnan(x)):
                df = wes.loc[cellline_name]
                gdsc_cell.add(cellline_name)
                gdsc_drug.add(drug_id)
                gdsc_data.append([int(drug_id), torch.tensor(df).float(), x])
                if x > 0:
                    p += 1
                else:
                    n += 1
    gdsc_cnv_dataset = LabeledDataset(gdsc_data)
    print('gdsc cell line:', len(gdsc_cell))
    print('gdsc drug:', len(gdsc_drug))
    print(p)
    print(n)
    with open(cwd + '/data/CNV/GDSC_cnv_dataset_z0_779_before.pkl','wb')as w2:
        pickle.dump(gdsc_cnv_dataset, w2)

    tcga_unlabel_data = []

    tcga_data = []
    for drug_id in TCGA_drug_list:
        for cellline_name in tcga_response.index:
            x = tcga_response.loc[cellline_name, drug_id]
            if cellline_name in tcga.index and (not np.isnan(x)):
                df = tcga.loc[cellline_name]
                tcga_data.append([int(drug_id), torch.tensor(df).float(), x])
    testDataset = LabeledDataset(tcga_data)
    with open(cwd + '/data/CNV/TCGA_cnv_test_dataset_z0_779_before.pkl','wb')as w:
        pickle.dump(testDataset, w)

    for sample in tcga.index:
        if sample not in test_set_patients:
            df = tcga.loc[sample]
            tcga_unlabel_data.append(df)
    tcga_unlabel_data = np.array(tcga_unlabel_data)
    tcga_unlabel_dataset = UnlabeledDataset(torch.tensor(tcga_unlabel_data).float())
    with open(cwd + '/data/CNV/TCGA_only_cnv_dataset_z0_779_before.pkl','wb')as w:
        pickle.dump(tcga_unlabel_dataset, w)

def makePDXDataset():
    gids = '6146,10243,2060,2064,2065,2066,2068,139285,83990,2071,2072,25,8216,27,8218,2078,10272,10274,6184,8233,8239,8241,8242,2099,8243,4149,2115,2118,2119,2120,2122,442444,10320,2130,2131,2132,53335,90,92,4193,2146,4194,8289,114788,8294,8295,10342,6249,53353,57448,8301,30835,4214,8312,8313,8314,4221,55422,2175,2176,2177,2178,2181,6278,84101,4233,114825,2188,2189,51340,2195,57492,2199,10397,4255,2213,4261,8358,51366,84133,2237,4286,2242,4291,4292,4297,4298,4299,4300,4301,4302,4303,207,208,55500,2260,2261,2262,2263,2264,217,8405,2271,2272,4330,238,6385,6389,6390,6391,57591,4352,10499,2308,2309,8452,84231,2313,2316,6416,2322,6418,2324,6421,22806,8471,6424,346389,6427,6428,286,55596,8493,6446,8496,90417,6455,51517,22849,324,57670,84295,10568,330,8522,57674,80204,26960,4436,6491,6495,6497,57698,355,55654,92521,367,369,80243,4478,387,51592,394,121227,399,29072,10644,405,84376,151963,4515,27044,10664,2475,29102,55728,80304,8626,80312,22978,340419,6597,6598,29126,8648,6602,8651,201163,6605,27086,463,6608,84433,466,471,472,2521,84441,476,57820,10721,2531,51684,23013,4582,63976,4585,492,10735,10736,4595,27125,4602,80380,4609,4610,6657,4613,4615,23048,283149,283150,4627,4629,8726,23067,545,546,4644,51755,23085,4654,10801,23092,567,4665,4666,6714,8764,2623,2624,2625,580,581,4683,595,596,602,604,605,607,608,340578,6756,613,8805,6760,64109,23152,23157,6774,6777,6778,639,641,256646,6794,10892,653,657,6801,23185,4763,668,2719,672,673,675,4771,4773,4780,4781,2735,8880,694,695,4791,4794,701,6850,221895,90827,2767,10962,8915,2776,2778,23261,8929,10978,6886,6887,4841,4849,4851,4853,4869,6917,776,23305,6926,6927,6929,6934,6935,6938,4893,811,27436,4913,4914,4916,11064,4921,4926,4928,833,353088,836,23365,64324,841,842,23373,2903,861,862,863,2909,865,2913,867,868,4958,7006,7015,11116,23405,90993,11122,7030,892,7037,894,896,898,84870,11143,7048,9098,2956,9101,9113,23451,11166,11168,23462,84902,7080,11177,940,9135,23476,5049,23484,11197,11200,7113,972,973,974,3021,5077,5079,7128,5081,9175,23512,5087,23522,5093,999,7150,1008,1009,9203,5108,7157,1015,1019,1021,1026,1027,7170,1029,7171,1031,7175,171017,3084,168975,171023,3091,3092,1045,1050,3105,5155,5156,3110,5159,23598,3131,11328,5187,23624,7248,7249,1106,1108,7253,3159,3169,9314,9321,3181,3195,3205,3207,3209,58508,29844,3227,3229,3237,25766,3239,5290,5291,5292,5295,9401,1213,3265,5324,1233,79058,1236,5335,54487,9444,7403,7409,3320,5371,64764,1277,3326,1280,1281,5378,7428,7430,3337,64783,64784,5395,5396,29974,7454,1316,79145,3371,7468,5424,5426,120114,5428,25913,30012,116028,7486,1345,7490,25925,3399,91464,5450,25937,7507,5460,7508,50515,3417,3418,7514,5468,1385,1387,7531,9582,7555,9611,9612,5518,64919,1436,1441,5537,85414,9639,5546,5551,26039,26040,50615,5566,26047,5573,5579,26065,54738,1496,83417,5594,1499,1501,3551,5604,5605,3558,9709,1523,3572,9715,3575,1540,54790,1558,7704,9774,63035,3646,7750,54855,3662,1616,140885,9817,1630,5727,5728,54880,3685,1639,1643,54894,1649,1654,1655,1656,3702,5753,7799,54904,124540,1662,3716,3717,3718,83596,3725,9869,5777,5781,5783,7832,5787,5788,5789,5796,54949,124583,7849,7852,9901,57007,3762,55294,81608,3791,9935,3799,79577,9950,3815,3817,7913,9967,9968,65268,5879,1785,57082,1788,5884,5885,5890,3845,5894,7942,5900,5903,10000,57105,79633,5910,10006,5914,57120,10019,5925,5927,3895,7994,3899,286530,94025,345930,8013,5966,57167,8019,143187,8021,1879,3927,5979,3932,8028,8030,8031,3936,5987,79718,6000,79728,92017,51059,55159,55160,3977,26511,8085,55193,8091,26524,10142,4000,1956,4004,4005,51119,196528,8115,1974,4026,10186,6092,1999,2000,6098,8148,2005,2006,55252,4066,4067,10215,51176,6125,2033,2034,8178,6134,4087,4088,4089,2042,2045,4094,51199'
    gids = gids.split(',')
    gid2name = {}
    
    with open(cwd + '/data/gene_name_id', 'r')as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().split(' ')
            gid2name[l[0]] = l[1]
    print(gid2name)

    replaced = {}
    with open(cwd + '/data/replace_gene_name', 'r')as f2:
        content = f2.readlines()
        for line in content:
            x = line.strip().split(';')
            res_name,replace_name = x
            replaced[res_name] = replace_name
    print(replaced)

    for key in gid2name.keys():
        name = gid2name[key]
        if name in replaced:
            gid2name[key] = replaced[name]

    result = []
    for gid in gids:
        if gid in gid2name:
            result.append(gid2name[gid])
        else:
            print('not fount '+gid)
    result_set = set(result)
    result_find = set()
    print(len(result))
    # drugs = ['paclitaxel', 'INC280', 'trastuzumab', 'BGJ398', 'LJM716', 'LEE011', 'CGM097', 'binimetinib-3.5mpk', 'LGH447', 'LDE225', 'dacarbazine', 'LLM871', 'abraxane', 'trametinib', 'CKX620', 'gemcitabine-50mpk', 'CLR457', 'erlotinib', 'binimetinib', 'INC424', 'BKM120', 'cetuximab', 'LKA136', 'LDK378', 'LGW813', 'untreated', 'BYL719', '5FU', 'encorafenib', 'TAS266', 'figitumumab"', 'LJC049', 'tamoxifen', 'LFA102', 'HSP990', 'HDM201', 'WNT974']

    data = pd.read_excel(cwd + '/data/PDX/pdx_gene_expr_original.xlsx', keep_default_na=False,index_col=0,header=0)
    # print(data)
    # 22665*399
    data_sum = data.sum(axis=0)
    data_expr_tpm = data.div(data_sum, axis=1) * 1e6 # fpkm -> tpm


    data_expr = np.log2(data_expr_tpm + 1e-21)

    dm = np.array(data_expr.mean(axis=1)).reshape(22665,1)
    m = np.tile(dm, (1,399))
    ds = np.array(data_expr.std(axis=1)).reshape(22665,1)
    s = np.tile(ds, (1,399))

    data_expr = (data_expr - m) / s
    data_expr = data_expr.loc[result]
    print(data_expr, dm, ds, data_expr.mean(axis=1), data_expr.std(axis=1))
    data_expr.to_csv(cwd + '/data/PDX/pdx_gene_expr.csv')

    data_response = pd.read_excel(cwd + '/data/PDX/pdx_response.xlsx',index_col=0,header=0)
    samples = list(data_expr.columns)
    data_response = data_response.loc[data_response.index.isin(samples)]
    data_response = data_response[['Treatment', 'Treatment type', 'ResponseCategory']]
    L = ['PD','SD','CR','PR']

    def conver(v):
        if v in ['PD', 'SD']:
            return 0
        else:
            return 1
        
    data_response = data_response[data_response['Treatment type'] == 'single']
    data_response = data_response[data_response['ResponseCategory'].isin(L)]
    data_response['ResponseCategory'] = data_response['ResponseCategory'].apply(conver)

    # drugs = list(set(data_response['Treatment']))
    # print(drugs)
    
    fe = dc.feat.ConvMolFeaturizer()
    # # 获取药物信息
    ok_drugs = {}
    with open(cwd + '/data/PDX/drugs.txt', 'r')as d:
        content = d.readlines()
        for line in content:
            if ';' in  line:
                l = line.strip().split(';')
                drug_name, smiles = l
                ok_drugs[drug_name] = smiles

                mol = Chem.MolFromSmiles(smiles)
                convmol = fe.featurize([mol])[0] # 特征矩阵

                # 获取邻接矩阵和特征矩阵
                adjacency_matrix = convmol.get_adjacency_list()
                k = adjacency_matrix
                k_ = np.zeros((len(k), len(k)))
                for i in range(len(k)):
                    for j in k[i]:
                        k_[i][j] = 1
                k = k_
                adjacency_matrix = k
                np.fill_diagonal(adjacency_matrix ,1)
                zeros = np.zeros((100,100))
                adj = np.array(adjacency_matrix)
                
                n = adj.shape[0]
                zeros[:n, :n] = adj
                fzeros = np.zeros((100,75))
                feature_matrix = np.array(convmol.get_atom_features())
                fzeros[:n,:] = feature_matrix
                # print(zeros.shape, fzeros.shape)
                np.savez(cwd + '/data/PDX/drug_graph/'+drug_name+'.npz', adj=zeros, feature = fzeros)

    ok_drugs = {}
    with open(cwd + '/data/PDX/drugs.txt', 'r')as d:
        content = d.readlines()
        for line in content:
            if ';' in  line:
                l = line.strip().split(';')
                name,smiles = l
                ok_drugs[name] = smiles

    data = []
    ds = list(ok_drugs.keys())
    res = data_response[data_response['Treatment'].isin(ds)]
    print(res) # 1315

    res1 = res[res['ResponseCategory'] == 1]  # 测试集正样本 68
    res0 = res[res['ResponseCategory'] == 0]  # 所有的负样本 1247
    print(res1, len(set(res1.index)))
    print(res0, len(set(res0.index)))

    indices_res_1 = res1.index
    res0_in = res0[res0.index.isin(indices_res_1)]  # 测试集负样本
    res0_out = res0[~res0.index.isin(indices_res_1)]  # 训练集，只需要expr

    data_train = []
    train_set = set()
    for row in data_expr.T.itertuples(index = True):
        pdx = row.Index
        if pdx not in train_set:
            pdx_expr = data_expr[pdx]
            pdx_expr = torch.tensor(pdx_expr.values, dtype=torch.float32)
            data_train.append(pdx_expr)
            train_set.add(pdx)

    # test
    data_test = []
    test_drugs = set()
    test_pdxs = set()
    pos = 0
    neg = 0
    for row in res0_in.itertuples(index=True):
        pdx = row.Index
        drug = row.Treatment
        label = row.ResponseCategory
        if label == 1:
            pos += 1
        else:
            neg += 1
        pdx_expr = data_expr[pdx]
        # print(pdx_expr)
        pdx_expr = torch.tensor(pdx_expr.values, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32).view(1)
        # print(label)
        data_test.append([drug, pdx_expr, label])
        test_drugs.add(drug)
        test_pdxs.add(pdx)

    for row in res1.itertuples(index=True):
        pdx = row.Index
        drug = row.Treatment
        label = row.ResponseCategory
        if label == 1:
            pos += 1
        else:
            neg += 1
        pdx_expr = data_expr[pdx]
        # print(pdx_expr)
        pdx_expr = torch.tensor(pdx_expr.values, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32).view(1)
        # print(label)
        data_test.append([drug, pdx_expr, label])
        test_drugs.add(drug)
        test_pdxs.add(pdx)


    pdx_train_dataset = UnlabeledDataset(data_train)
    pdx_test_dataset = LabeledDataset(data_test)
    with open(cwd + '/data/PDX/pdx_train_dataset.pkl', 'wb')as s:
        pickle.dump(pdx_train_dataset, s)
    with open(cwd + '/data/PDX/pdx_test_dataset.pkl', 'wb')as w:
        pickle.dump(pdx_test_dataset, w)
    print('无标签pdx训练集条目:',len(pdx_train_dataset))
    print('pdx测试集条目:', len(pdx_test_dataset))

    print(len(train_set))
    print('测试集药物数:', len(test_drugs))
    print('测试集pdx数:', len(test_pdxs))
    print('测试集正样本数:', pos)
    print('测试集负样本数:', neg)



if __name__ == '__main__':
    pass
