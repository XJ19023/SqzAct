'''
(batch, Ni) x (Ni, No)
(batch, IC) x (IC, OC)
writes ==> buffer

base: W4A8
'''
from re import I
import pandas
import configparser
import os
import numpy as np
import bitfusion.src.benchmarks.benchmarks as benchmarks
from bitfusion.src.simulator.stats import Stats
from bitfusion.src.simulator.simulator import Simulator
from bitfusion.src.sweep.sweep import SimulatorSweep, check_pandas_or_run
from bitfusion.src.utils.utils import *
from bitfusion.src.optimizer.optimizer import optimize_for_order, get_stats_fast
from bitfusion.src.simulator.globalVar import set_run_mode, set_hw_arch

def df_to_stats(df):
    stats = Stats()
    stats.total_cycles = float(df['Cycles'].iloc[0])
    stats.mem_stall_cycles = float(df['Memory wait cycles'].iloc[0])
    stats.reads['act'] = float(df['IBUF Read'].iloc[0])
    stats.reads['out'] = float(df['OBUF Read'].iloc[0])
    stats.reads['wgt'] = float(df['WBUF Read'].iloc[0])
    stats.reads['dram'] = float(df['DRAM Read'].iloc[0])
    stats.writes['act'] = float(df['IBUF Write'].iloc[0])
    stats.writes['out'] = float(df['OBUF Write'].iloc[0])
    stats.writes['wgt'] = float(df['WBUF Write'].iloc[0])
    stats.writes['dram'] = float(df['DRAM Write'].iloc[0])
    return stats

sim_sweep_columns = ['N', 'M',
        'Max Precision (bits)', 'Min Precision (bits)',
        'Network', 'Layer',
        'Cycles', 'Memory wait cycles',
        'WBUF Read', 'WBUF Write',
        'OBUF Read', 'OBUF Write',
        'IBUF Read', 'IBUF Write',
        'DRAM Read', 'DRAM Write',
        'Bandwidth (bits/cycle)',
        'WBUF Size (bits)', 'OBUF Size (bits)', 'IBUF Size (bits)',
        'Batch size']

batch_size = 64

results_dir = './results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open('123.log', 'w') as f:
    f.writelines(f'\n')

# '''
# qwt
#  base Mant
set_run_mode('base')
set_hw_arch('mant')
config_file = 'hardwareConf/mant.ini'
# Create simulator object
bf_e_sim = Simulator(config_file, False)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'mant.csv')
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type='mant')
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
bf_e_cycles_mant = []
bf_e_energy_mant = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_mant.append(bf_e_stats.total_cycles)
    bf_e_energy_mant.append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))
with open('123.log', 'a') as f:
    f.writelines(f'>>> Mant: {bf_e_stats}\n\n')
# '''

# Olive
set_run_mode('base')
set_hw_arch('olive')
config_file = 'hardwareConf/olive.ini'
# Create simulator object
bf_e_sim = Simulator(config_file, False)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'olive.csv')
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type='olive')
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
bf_e_cycles_olive = []
bf_e_energy_olive = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_olive.append(bf_e_stats.total_cycles)
    bf_e_energy_olive.append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))
with open('123.log', 'a') as f:
    f.writelines(f'>>> Olive: {bf_e_stats}\n')

'''
all_energy1 = []
for i in range(len(bf_e_cycles_mant)):
    energy_data_1 = bf_e_energy_mant[i]
    energy_data_total = energy_data_1[0] + energy_data_1[1] + energy_data_1[2] + energy_data_1[3]
    energy_data_1[0] /= energy_data_total
    energy_data_1[1] /= energy_data_total
    energy_data_1[2] /= energy_data_total
    energy_data_1[3] /= energy_data_total
print()
for i in energy_data_1:
    print("%0.4f, " %(i), end="")

exit()
'''

# Spark
set_run_mode('base')
set_hw_arch('spark')
config_file = 'hardwareConf/spark.ini'
# Create simulator object
bf_e_sim = Simulator(config_file, False)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'spark.csv')
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type='spark')
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
bf_e_cycles_spark = []
bf_e_energy_spark = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_spark.append(bf_e_stats.total_cycles)
    bf_e_energy_spark.append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))
with open('123.log', 'a') as f:
    f.writelines(f'>>> Spark: {bf_e_stats}\n')

'''
# qwt no reorder
set_run_mode('qwt')
set_hw_arch('squeeze')
config_file = 'hardwareConf/squeezeAct.ini'
# Create simulator object
bf_e_sim = Simulator(config_file, False)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'ant_os.csv')
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type='qwt')
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
bf_e_cycles_squeeze = []
bf_e_energy_squeeze = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_squeeze.append(bf_e_stats.total_cycles)
    bf_e_energy_squeeze.append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))
with open('123.log', 'a') as f:
    f.writelines(f'>>> qwt: {bf_e_stats}\n')
'''

# qwt reordered
set_run_mode('reorder')
set_hw_arch('squeeze')
config_file = 'hardwareConf/squeezeAct.ini'
# Create simulator object
bf_e_sim = Simulator(config_file, False)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'squeeze.csv')
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type='squeeze')
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
bf_e_cycles_squeeze = []
bf_e_energy_squeeze = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_squeeze.append(bf_e_stats.total_cycles)
    bf_e_energy_squeeze.append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))
with open('123.log', 'a') as f:
    f.writelines(f'>>> squeeze: {bf_e_stats}\n\n')

all_cyc = []
cyc_1_mean = 0
cyc_2_mean = 0
cyc_3_mean = 0
cyc_4_mean = 0
cyc_baseline_mean = 0

# write to csv
model_name_dict = {'vgg16':'VGG16', 
                   'resnet18':'ResNet18',
                   'resnet50':'ResNet50',
                   'inceptionv3':'InceptionV3',
                   'vit':'ViT',
                   'mnli':'BERT-MNLI',
                   'cola':'BERT-CoLA',
                   'sst_2':'BERT-SST-2',
                   'wikitext_TinyLlama_1_1B_Chat_v1_0':'Llama1-1B',
                   'wikitext_llama_2_7b_hf':'llama2-7B',
                   'wikitext_Meta_Llama_3_8B':'Llama3-8B',
                   'wikitext_Qwen2_5_0_5B':'Qwen2.5-0.5B',
                   'wikitext_Qwen2_5_1_5B':'Qwen2.5-1.5B',
                   'wikitext_Qwen2_5_7B':'Qwen2.5-7B',
                   }

ff = open(os.getcwd() + '/results/my_res.csv', "a")
wr_line = "Time, "
wr_bench_name = ", "
wr_model_name = ", "
print()
for i in range(len(bf_e_cycles_mant)):
    model_name = benchmarks.benchlist[i]

    cyc_baseline = bf_e_cycles_mant[i]
    cyc_1 = bf_e_cycles_mant[i] / cyc_baseline
    cyc_1_mean += cyc_1
    cyc_2 = bf_e_cycles_olive[i] / cyc_baseline
    cyc_2_mean += cyc_2
    cyc_3 = bf_e_cycles_spark[i] / cyc_baseline
    cyc_3_mean += cyc_3
    cyc_4 = bf_e_cycles_squeeze[i] / cyc_baseline
    cyc_4_mean += cyc_4
    cyc_baseline = cyc_baseline / cyc_baseline
    cyc_baseline_mean += cyc_baseline
    
    # all_cyc.append(cyc_1)
    all_cyc.append(cyc_baseline)
    all_cyc.append(cyc_2)
    all_cyc.append(cyc_3)
    all_cyc.append(cyc_4)

    wr_model_name += model_name_dict[model_name] + ", , , ,"
    wr_bench_name += "MANT, OliVe, SPARK, SqueezeAct, "
    wr_line += "%0.4f, %0.4f, %0.4f, %0.4f, " %(cyc_1, cyc_2, cyc_3, cyc_4)
    print("%0.4f, %0.4f, %0.4f, %0.4f, " %(cyc_1, cyc_2, cyc_3, cyc_4), end="")

cyc_1_mean /= len(bf_e_cycles_mant)
cyc_2_mean /= len(bf_e_cycles_mant)
cyc_3_mean /= len(bf_e_cycles_mant)
cyc_4_mean /= len(bf_e_cycles_mant)

wr_model_name += "Geomean, , , , \n"
wr_bench_name += "MANT, OliVe, SPARK, Squeeze, \n"
wr_line += ("%0.4f, %0.4f, %0.4f, %0.4f, " %(cyc_1_mean, cyc_2_mean, cyc_3_mean, cyc_4_mean)) + "\n\n"
ff.write(wr_model_name)
ff.write(wr_bench_name)
ff.write(wr_line)
wr_line = ""
print("%0.4f, %0.4f, %0.4f, %0.4f, " %(cyc_1_mean, cyc_2_mean, cyc_3_mean, cyc_4_mean))

all_energy1 = []
all_energy2 = []
all_energy3 = []
all_energy4 = []
for i in range(len(bf_e_cycles_mant)):

    model_name = benchmarks.benchlist[i]

    energy_data_1 = bf_e_energy_mant[i]
    energy_data_total = energy_data_1[0] + energy_data_1[1] + energy_data_1[2] + energy_data_1[3]

    energy_data_1[0] /= energy_data_total
    energy_data_1[1] /= energy_data_total
    energy_data_1[2] /= energy_data_total
    energy_data_1[3] /= energy_data_total

    energy_data_2 = bf_e_energy_olive[i]
    energy_data_2[0] /= energy_data_total
    energy_data_2[1] /= energy_data_total
    energy_data_2[2] /= energy_data_total
    energy_data_2[3] /= energy_data_total

    energy_data_3 = bf_e_energy_spark[i]
    energy_data_3[0] /= energy_data_total
    energy_data_3[1] /= energy_data_total
    energy_data_3[2] /= energy_data_total
    energy_data_3[3] /= energy_data_total

    energy_data_4 = bf_e_energy_squeeze[i]
    energy_data_4[0] /= energy_data_total
    energy_data_4[1] /= energy_data_total
    energy_data_4[2] /= energy_data_total
    energy_data_4[3] /= energy_data_total

    all_energy1.append(energy_data_1[0])
    all_energy1.append(energy_data_2[0])
    all_energy1.append(energy_data_3[0])
    all_energy1.append(energy_data_4[0])

    all_energy2.append(energy_data_1[1])
    all_energy2.append(energy_data_2[1])
    all_energy2.append(energy_data_3[1])
    all_energy2.append(energy_data_4[1])

    all_energy3.append(energy_data_1[2])
    all_energy3.append(energy_data_2[2])
    all_energy3.append(energy_data_3[2])
    all_energy3.append(energy_data_4[2])

    all_energy4.append(energy_data_1[3])
    all_energy4.append(energy_data_2[3])
    all_energy4.append(energy_data_3[3])
    all_energy4.append(energy_data_4[3])

print()

wr_line = "Static, "
for i in all_energy1:
    wr_line += "%0.4f, " %(i)
    print("%0.4f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0

for i in range(len(bf_e_cycles_mant)):
    model_name = benchmarks.benchlist[i]
    idx = i * 4
    energy_mean_1 += all_energy1[idx]
    energy_mean_2 += all_energy1[idx+1]
    energy_mean_3 += all_energy1[idx+2]
    energy_mean_4 += all_energy1[idx+3]

energy_mean_1 /= len(bf_e_cycles_mant)
energy_mean_2 /= len(bf_e_cycles_mant)
energy_mean_3 /= len(bf_e_cycles_mant)
energy_mean_4 /= len(bf_e_cycles_mant)

wr_line += ("%0.4f, %0.4f, %0.4f, %0.4f, " %(energy_mean_1, energy_mean_2, energy_mean_3, energy_mean_4)) + "\n"
ff.write(wr_model_name)
ff.write(wr_bench_name)
ff.write(wr_line)
wr_line = ""
print("%0.4f, %0.4f, %0.4f, %0.4f, " %(energy_mean_1, energy_mean_2, energy_mean_3, energy_mean_4))

wr_line = "Dram, "
for i in all_energy2:
    wr_line += "%0.4f, " %(i)
    print("%0.4f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0

for i in range(len(bf_e_cycles_mant)):
    model_name = benchmarks.benchlist[i]
    idx = i * 4
    energy_mean_1 += all_energy2[idx]
    energy_mean_2 += all_energy2[idx+1]
    energy_mean_3 += all_energy2[idx+2]
    energy_mean_4 += all_energy2[idx+3]

energy_mean_1 /= len(bf_e_cycles_mant)
energy_mean_2 /= len(bf_e_cycles_mant)
energy_mean_3 /= len(bf_e_cycles_mant)
energy_mean_4 /= len(bf_e_cycles_mant)

wr_line += ("%0.4f, %0.4f, %0.4f, %0.4f," %(energy_mean_1, energy_mean_2, energy_mean_3, energy_mean_4)) + "\n"
ff.write(wr_line)
wr_line = ""
print("%0.4f, %0.4f, %0.4f, %0.4f," %(energy_mean_1, energy_mean_2, energy_mean_3, energy_mean_4))

wr_line = "Buffer, "
for i in all_energy3:
    wr_line += "%0.4f, " %(i)
    print("%0.4f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0

for i in range(len(bf_e_cycles_mant)):
    model_name = benchmarks.benchlist[i]
    idx = i * 4
    energy_mean_1 += all_energy3[idx]
    energy_mean_2 += all_energy3[idx+1]
    energy_mean_3 += all_energy3[idx+2]
    energy_mean_4 += all_energy3[idx+3]

energy_mean_1 /= len(bf_e_cycles_mant)
energy_mean_2 /= len(bf_e_cycles_mant)
energy_mean_3 /= len(bf_e_cycles_mant)
energy_mean_4 /= len(bf_e_cycles_mant)

wr_line += ("%0.4f, %0.4f, %0.4f, %0.4f," %(energy_mean_1, energy_mean_2, energy_mean_3, energy_mean_4)) + "\n"
ff.write(wr_line)
wr_line = ""
print("%0.4f, %0.4f, %0.4f, %0.4f," %(energy_mean_1, energy_mean_2, energy_mean_3, energy_mean_4))

wr_line = "Core, "
for i in all_energy4:
    wr_line += "%0.4f, " %(i)
    print("%0.4f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0

for i in range(len(bf_e_cycles_mant)):
    model_name = benchmarks.benchlist[i]
    idx = i * 4
    energy_mean_1 += all_energy4[idx]
    energy_mean_2 += all_energy4[idx+1]
    energy_mean_3 += all_energy4[idx+2]
    energy_mean_4 += all_energy4[idx+3]

energy_mean_1 /= len(bf_e_cycles_mant)
energy_mean_2 /= len(bf_e_cycles_mant)
energy_mean_3 /= len(bf_e_cycles_mant)
energy_mean_4 /= len(bf_e_cycles_mant)

wr_line += ("%0.4f, %0.4f, %0.4f, %0.4f," %(energy_mean_1, energy_mean_2, energy_mean_3, energy_mean_4)) + "\n\n"
ff.write(wr_line)
wr_line = ""
print("%0.4f, %0.4f, %0.4f, %0.4f," %(energy_mean_1, energy_mean_2, energy_mean_3, energy_mean_4))

print("Please see the results at ./results/my_res.csv ")
ff.close()
