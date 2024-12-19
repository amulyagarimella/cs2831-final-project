# Original Author: Shawn Bai
# Substantial edits + hypothesis tests: Amulya Garimella
# Last edits: 2024-11-19
## improved saving of bed files - integer start and end
## saved bedgraph
## improved window 

import cooler
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Import the time module
import sys
import subprocess
import typer
from typing import List, Optional
from typing_extensions import Annotated
import os
from pathlib import Path
from datetime import datetime
import re
import multiprocessing
import pybedtools

app = typer.Typer()

def current_timestamp ():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def dir_current_timestamp (dir_name="", basedir_name=None, makedir=True):
    ct = current_timestamp()
    if basedir_name is not None:
        basedir = Path(basedir_name).resolve()
    else:
        basedir = Path.cwd()
    if len(dir_name) > 0:
        dir_name = "_" + dir_name
    outpath = ( basedir / f"{ct}{dir_name}" ).resolve()
    if makedir:
        outpath.mkdir(parents=True)
    return outpath
    

# Load the .cool file and define the region of interest on chromosome 
class PotentialStripes:
    def __init__ (self, 
                  clr_path, 
                  chrom_sizes_path,
                  cores):
        self.clr = cooler.Cooler(clr_path)
        #self.chrom_sizes = pd.read_csv(chrom_sizes_path, sep='\t', header=None, names=['chrom', 'size'])
        self.chrom_sizes = pd.DataFrame(self.clr.chromsizes).reset_index()
        self.chrom_sizes.columns = ['chrom', 'size']
        print(self.chrom_sizes)
        self.resol = self.clr.binsize
        self.cores = cores
        use_chr = self.clr.chromnames[0][0] == 'c'
        print("use chr", use_chr)
        use_chr_chrsizes = self.chrom_sizes['chrom'].values[0][0] == 'c'
        print("use chr chrsizes", use_chr_chrsizes)
        self.chr = 'chr' if use_chr else ''
        if use_chr and not use_chr_chrsizes:
            self.chrom_sizes['chrom'] = self.chr + self.chrom_sizes['chrom']
        elif not use_chr and use_chr_chrsizes:
            print("replacing chr")
            self.chrom_sizes['chrom'] = self.chrom_sizes['chrom'].str.replace("chr", "")

    # Function to compute interaction sums for a given row index
    def _compute_interaction_sum(self, i, window_size, matrix):
        start_index = max(0, i - window_size)
        end_index = min(len(matrix), i + window_size + 1)
        # Exclude the diagonal value by summing the upper and lower parts separately
        sum_upper = np.nansum(matrix[start_index:i-2, i])
        sum_lower = np.nansum(matrix[i+2:end_index, i])
        strand = '+' if sum_upper > sum_lower else '-'
        return sum_upper + sum_lower, sum_upper, sum_lower, strand
    
    # TODO sliding window pval calc
    def calculate_multiscale_lambda (self, chromosome, peak_center_idx, sum_uppers, sum_lowers, peak_center_coord, window_sizes):
        window_sizes = sorted(np.array(window_sizes))

        if peak_center_idx < window_sizes[-1]//2//self.clr.binsize or peak_center_idx > len(sum_uppers) - window_sizes[-1]//2//self.clr.binsize:
            bound1 = max(peak_center_coord - window_sizes[-1]//2, 0)
            bound2 = min(peak_center_coord + window_sizes[-1]//2, self.chrom_sizes[self.chrom_sizes['chrom'] == chromosome]['size'].values[0])
            try:
                matrix = self.clr.matrix(balance=False).fetch(f"{chromosome}:{bound1}-{bound2}")
                interaction_sums_and_strands = [self._compute_interaction_sum(i, window_sizes[-1], matrix) for i in range(len(matrix))]
                sum_uppers = np.array([item[1] for item in interaction_sums_and_strands])
                sum_lowers = np.array([item[2] for item in interaction_sums_and_strands])
                peak_center_idx = len(matrix)//2
            except:
                print(f"Peak {peak_center_coord} is too close to the edge of the chromosome. Skipping.")
                print(f"Peak center index: {peak_center_idx}, window size: {window_sizes[-1]}, bound 1: {bound1}, bound 2: {bound2}")
                return [], [], [], [], window_sizes
        
        mean_upper_signals = []
        std_dev_upper_signals = []
        mean_lower_signals = []
        std_dev_lower_signals = []
        for window_size in window_sizes:
            start_index = peak_center_idx - window_size//2//self.clr.binsize
            end_index = peak_center_idx + window_size//2//self.clr.binsize + 1
            # print(start_index, end_index)
            sum_upper = sum_uppers[start_index:end_index]
            sum_lower = sum_lowers[start_index:end_index]
            mean_upper_signals.append(np.mean(sum_upper))
            std_dev_upper_signals.append(np.std(sum_upper))
            mean_lower_signals.append(np.mean(sum_lower))
            std_dev_lower_signals.append(np.std(sum_lower))
            
        return mean_upper_signals, std_dev_upper_signals, mean_lower_signals, std_dev_lower_signals, window_sizes

    def find_peaks_chrom (self, chromosome, chrom_size, peak_calling_window_size, peak_calling_threshold, large_window_size, large_window_step_size, sum_window_size, lambda_window_sizes, save):
        start_time = time.time()
        start = 0
        bed_df_all = []
        bed_df_thresh = []
        while start < chrom_size:
            end = min(start + large_window_size, chrom_size)
            region = f"{chromosome}:{start}-{end}"
            print(region)
            matrix = self.clr.matrix(balance=False).fetch(region)
            chromosome, positions = region.split(':')
            
            # Calculate interaction sums and strands separately
            interaction_sums_and_strands = [self._compute_interaction_sum(i, sum_window_size, matrix) for i in range(len(matrix))]
            interaction_sums = np.array([item[0] for item in interaction_sums_and_strands])  # Numeric data for statistical calculations

            sum_uppers = np.array([item[1] for item in interaction_sums_and_strands])
            sum_lowers = np.array([item[2] for item in interaction_sums_and_strands])
            strands = [item[3] for item in interaction_sums_and_strands]

            # # Convert the list to a NumPy array
            # sum_signals = np.array(interaction_sums)
            # Ensure interaction_sums is not empty before calculating statistical properties
            if interaction_sums.size > 0:
                # Calculate genomic coordinates and output the detailed information for each bin
                region_start = int(start)
                bins = range(len(matrix))
                bin_starts = [region_start + (bin * self.clr.binsize) for bin in bins]
                bin_ends = [start + self.clr.binsize for start in bin_starts]  # Assuming each bin represents a self.clr.binsize bp region
                #strands_for_bins = [strand for strand in strands]  # Already computed strands for each bin
                #sum_uppers_for_bins = [sum_upper for sum_upper in sum_uppers]  # sum_upper for each bin
                #sum_lowers_for_bins = [sum_lower for sum_lower in sum_lowers]  # sum_lower for each bin

                # Create BED format DataFrame for all bins
                bed_df_all_bins = pd.DataFrame({
                    'chrom': [chromosome] * len(matrix),
                    'start': bin_starts,
                    'end': bin_ends,
                    'strand' : strands,
                    'sum_upper': sum_uppers,
                    'sum_lower': sum_lowers
                })
                bed_df_all.append(bed_df_all_bins)
            else:
                print("No interaction sums were calculated. Check your matrix and window size.")
            start += large_window_step_size
        allpeaks = pd.concat(bed_df_all)
        allpeaks = allpeaks.groupby(['chrom', 'start', 'end']).max().reset_index()
        if len(save) > 0:
            allpeaks.to_csv(f"{save}/allpeaks_{chromosome}.bed", sep='\t', index=False)
            # TODO
        start = 0

        while start < chrom_size:
            end = min(start + large_window_size, chrom_size)
            region = allpeaks.loc[(allpeaks['start'] >= start) & (allpeaks['end'] <= end)]
            interaction_sums = np.array(region['sum_upper'] + region['sum_lower'])  
            # Find peaks
            if interaction_sums.size > 0:
                mean_signal = np.mean(interaction_sums)
                std_dev_signal = np.std(interaction_sums)
                cutoff_value = mean_signal + std_dev_signal * peak_calling_threshold  
                print(f"Mean signal: {mean_signal}, Standard deviation: {std_dev_signal}, Cutoff value: {cutoff_value}")

                peaks, _ = find_peaks(interaction_sums, wlen=peak_calling_window_size, prominence=cutoff_value)
                print(f"Number of peaks found: {len(peaks)}")

                # Calculate genomic coordinates and output the peak information
                peak_coordinates = [start + (peak * self.clr.binsize) for peak in peaks]
                interaction_sums = np.array(region['sum_upper'] + region['sum_lower'])  
         
                p_uppers = []
                p_lowers = []
                comp_lambda_uppers = []
                comp_lambda_lowers = []

                peaks_df = region.iloc[peaks,:]
                upper = list(peaks_df.loc[:,'sum_upper'])
                lower = list(peaks_df.loc[:,'sum_lower'])

                for i in range(len(peaks)):
                    mean_upper_signals, _, mean_lower_signals, _, _ = self.calculate_multiscale_lambda(chromosome, peaks[i], sum_uppers, sum_lowers, peak_coordinates[i], lambda_window_sizes)
                    if len(mean_upper_signals) == 0:
                        comp_lambda_uppers.append(np.nan)
                        comp_lambda_lowers.append(np.nan)
                        p_uppers.append(np.nan)
                        p_lowers.append(np.nan)
                        continue
                    comp_lambda_upper = np.max(mean_upper_signals)
                    comp_lambda_uppers.append(comp_lambda_upper)
                    comp_lambda_lower = np.max(mean_lower_signals)
                    comp_lambda_lowers.append(comp_lambda_lower)
                    p_upper = stats.poisson.sf(upper[i], comp_lambda_upper)
                    p_uppers.append(p_upper)
                    print(f"DEBUG - window signals: {mean_upper_signals} vs. {upper[i]} -> p = {p_upper}")
                    p_lower = stats.poisson.sf(lower[i], comp_lambda_lower)
                    p_lowers.append(p_lower)

                peaks_df = peaks_df.assign(compLambdaUpper=comp_lambda_uppers, compLambdaLower=comp_lambda_lowers, upperP=p_uppers, lowerP=p_lowers)
                peaks_df['start'] =  peaks_df['start'].astype(int)
                peaks_df['end'] =  peaks_df['end'].astype(int)
                bed_df_thresh.append(peaks_df)
                print(peaks_df.head())
                print(f"Peak coordinates calculated.")
            else:
                print("No interaction sums were calculated. Check your matrix and window size.")
            start += large_window_size 

        peaks = pd.concat(bed_df_thresh)
        time_taken = time.time() - start_time
        if len(save) > 0:
            peaks.to_csv(f"{save}/threshpeaks_{chromosome}.bed", sep='\t', index=False)
        return allpeaks, peaks, time_taken

    # Function to find peaks
    def find_peaks (self, chrom, peak_calling_window_size, peak_calling_threshold, large_window_size, large_window_step_size, sum_window_size, lambda_window_sizes, save):
        # chrom = [str(c).replace("chr","") for c in chrom]
        all_chrom_allpeaks = []
        all_chrom_thresh_peaks = []
        all_chrom_times_taken = []

        with multiprocessing.Pool(processes=self.cores) as pool:
            results = [pool.apply_async(self.find_peaks_chrom, args=(chromosome, chrom_size, peak_calling_window_size, peak_calling_threshold, large_window_size, large_window_step_size, sum_window_size, lambda_window_sizes, save)) for chromosome, chrom_size in zip(self.chrom_sizes["chrom"], self.chrom_sizes["size"]) if chromosome in chrom]
            for result in results:
                chrom_allpeaks, chrom_peaks, chrom_time_taken = result.get()
                all_chrom_allpeaks.append(chrom_allpeaks)
                all_chrom_thresh_peaks.append(chrom_peaks)
                all_chrom_times_taken.append(chrom_time_taken)
        
        # Concatenate all dataframes
        all_chrom_allpeaks_df = pd.concat(all_chrom_allpeaks)
        all_chrom_thresh_peaks_df = pd.concat(all_chrom_thresh_peaks)
        return all_chrom_allpeaks_df, all_chrom_thresh_peaks_df, all_chrom_times_taken

def main (input_clr : Annotated[Optional[str], typer.Option()] = '/Users/amulyagarimella/Desktop/Xia_Lab/J/data/HCT116/HCT_WT_new_merged.pairs_dedup_941M_200_orig.mcool::/resolutions/400',
          chrom_sizes : Annotated[Optional[str], typer.Option()] = '/Users/amulyagarimella/genome/hg38/hg38.chrom.sizes',
          chrom_num : Annotated[Optional[List[str]], typer.Option()] = [], 
          peak_calling_threshold : Annotated[Optional[float], typer.Option()] = 1.0, 
          peak_calling_window_size : Annotated[Optional[int], typer.Option()] = None,
          interaction_sum_large_window_size : Annotated[Optional[int], typer.Option()]= 20000000, 
          interaction_sum_large_window_step_size : Annotated[Optional[int], typer.Option()] = None, 
          interaction_sum_window_size : Annotated[Optional[int], typer.Option()]= 1000000, 
          lambda_window_sizes : Annotated[Optional[List[int]], typer.Option()] = [10000,100000,1000000],
          p_thresholds : Annotated[Optional[List[float]], typer.Option()] = [0.05, 0.01, 5e-8], 
          n_cores : Annotated[Optional[int], typer.Option()] = 4,
          out_prefix : Annotated[Optional[str], typer.Option()] = "/Users/amulyagarimella/Desktop/Xia_Lab/J/outputs",
          save_indiv_chr : Annotated[Optional[bool], typer.Option()] = True):
    
    # Create output directory if it doesn't exist
    interaction_sum_large_window_step_size = interaction_sum_large_window_size - interaction_sum_window_size if interaction_sum_large_window_step_size is None else interaction_sum_large_window_step_size

    peak_calling_window_size = peak_calling_window_size if peak_calling_window_size is not None else interaction_sum_window_size//2
    chrom_num_outsuffix = "_".join(chrom_num) if len(chrom_num) > 0 else "all"
    ps = PotentialStripes(clr_path=input_clr, chrom_sizes_path=chrom_sizes, cores=n_cores)

    outsuffix = f"resolution_{ps.resol}bp_windowsize_{interaction_sum_large_window_size//1000000}Mb_stepsize_{interaction_sum_large_window_step_size//1000000}Mb_sumwindowsize_{interaction_sum_window_size//1000}kb_peakwindowsize_{peak_calling_window_size//1000}kb_threshold_{peak_calling_threshold}_lambdawindowsizesbp_{'_'.join([str(i) for i in lambda_window_sizes])}_chromosomes_{chrom_num_outsuffix}"

    output_dir = str(dir_current_timestamp(dir_name=f"stripe_prior_results_{outsuffix}", basedir_name=out_prefix, makedir=True))
    
    log_file = f"{output_dir}/parameters.log"
    with open(log_file, 'w') as f:
        f.write(f"Input .cool file: {input_clr}\n")
        f.write(f"Resolution: {ps.resol}\n")
        f.write(f"Chromosome sizes file: {chrom_sizes}\n")
        f.write(f"Threshold: {peak_calling_threshold}\n")
        f.write(f"Large window - size: {interaction_sum_large_window_size}\n")
        f.write(f"Large window - step size: {interaction_sum_large_window_step_size}\n")
        f.write(f"Summation window size: {interaction_sum_window_size}\n")
        f.write(f"Peak calling window size: {peak_calling_window_size}\n")
        f.write(f"Chromosome(s): {chrom_num}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Lambda window sizes: {lambda_window_sizes}\n")

    start_time = time.time()
    
    if len(chrom_num) == 0:
        chrom = ps.clr.chromnames
    else:
        chrom = [f"{ps.chr}{c}" for c in chrom_num]
    
    all_chrom_allpeaks_df, all_chrom_thresh_peaks_df, times_taken = ps.find_peaks(chrom, peak_calling_window_size = int(peak_calling_window_size), peak_calling_threshold = float(peak_calling_threshold), large_window_size = int(interaction_sum_large_window_size), large_window_step_size = int(interaction_sum_large_window_step_size), sum_window_size=int(interaction_sum_window_size), lambda_window_sizes=lambda_window_sizes, save=output_dir if save_indiv_chr else "")

    # Write output to BED files
    outfile_all = f"{output_dir}/prior_peaks_all_chromosomes_.bed"
    outfile_th = f"{output_dir}/prior_peaks_all_chromosomes_peak_calling_threshold_{peak_calling_threshold}_{outsuffix}.bed"
    all_chrom_allpeaks_df.to_csv(outfile_all, sep='\t', index=False, header=True)
    all_chrom_thresh_peaks_df['start'] =  all_chrom_thresh_peaks_df['start'].astype(int)
    all_chrom_thresh_peaks_df['end'] =  all_chrom_thresh_peaks_df['end'].astype(int)
    all_chrom_thresh_peaks_df.to_csv(outfile_th, sep='\t', index=False, header=True)

    upper = f"{output_dir}/prior_peaks_all_chromosomes_{outsuffix}.upper"

    lower = f"{output_dir}/prior_peaks_all_chromosomes_{outsuffix}.lower"

    chrom_sizes_dict =  {f"{k}" : (1,v) for k, v in dict(ps.clr.chromsizes).items()}
    upper_uncut = pybedtools.BedTool.from_dataframe(all_chrom_allpeaks_df.loc[:,['chrom','start','end','sum_upper']])
    lower_uncut = pybedtools.BedTool.from_dataframe(all_chrom_allpeaks_df.loc[:,['chrom','start','end','sum_lower']])
    upper_cut = upper_uncut.slop(b=0,g=chrom_sizes_dict).to_dataframe()
    lower_cut = lower_uncut.slop(b=0,g=chrom_sizes_dict).to_dataframe()

    lower_cut_for_viz = lower_cut.copy(deep=True)
    lower_cut_for_viz['name'] = -1*lower_cut_for_viz['name']
    
    upper_cut.to_csv(f"{upper}.bedGraph", sep='\t', index=False, header=False)
    lower_cut.to_csv(f"{lower}.bedGraph", sep='\t', index=False, header=False)
    lower_cut_for_viz.to_csv(f"{lower}.viz.bedGraph", sep='\t', index=False, header=False)

    subprocess.run(f"bedGraphToBigWig {upper}.bedGraph {chrom_sizes} {upper}.bw", shell=True, text=True, check=True)
    subprocess.run(f"bedGraphToBigWig {lower}.viz.bedGraph {chrom_sizes} {lower}.viz.bw", shell=True, text=True, check=True)

    for p in p_thresholds:
        outfile_p = f"{output_dir}/prior_peaks_all_chromosomes_peak_calling_threshold_{peak_calling_threshold}_{outsuffix}_p_{p}.bed"
        all_chrom_thresh_peaks_df[(all_chrom_thresh_peaks_df['upperP'] < p) | (all_chrom_thresh_peaks_df['lowerP'] < p)].to_csv(outfile_p, sep='\t', index=False, header=False)

    # Write parameters to log file
    #log_file = f"{output_dir}/parameters.log"
    with open(log_file, 'a') as f:
        f.write(f"--- {(time.time() - start_time)} seconds taken ---\n")
        for i in range(len(times_taken)):
            f.write(f"Time taken for chromosome {chrom[i]}: {times_taken[i]}\n")
    
if __name__ == "__main__":
    typer.run(main)