import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import ImageProcessing

matplotlib.use('pdf')
import statistics as stat
from skimage import feature

import time
import random
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import warnings
from scipy.ndimage import convolve
from matplotlib.markers import MarkerStyle
from scipy.stats import poisson


warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

class getStripe:
    def __init__(self, unbalLib, bed, signal, resol, minH, maxW, canny, all_chromnames, chromnames, all_chromsizes, chromsizes, core, bfilter, seed, windowSize, full_chromsizes, method, logfilename):
        self.unbalLib = unbalLib
        self.locations = bed
        self.signal = signal
        self.resol = resol
        self.minH = minH
        self.maxW = maxW
        self.canny = canny
        self.all_chromnames = all_chromnames
        """
        AG 2024-10-13: 
        self.all_chromsizes is a dictionary with form
        {chromname: [(0, chromsize)]}
        self.chromsizes is a dictionary with form
        {chromname: [(start_0, end_0), (start_1, end_1), ...]}
        """
        self.all_chromsizes = all_chromsizes
        self.chromnames = chromnames
        self.chromsizes = chromsizes 
        self.windowSize = windowSize
        self.core = core
        self.bfilter = bfilter
        self.prng = random.Random(seed)
        """self.chromnames2sizes={}
        for i in range(len(self.all_chromnames)):
            self.chromnames2sizes[self.all_chromnames[i]] = self.all_chromsizes[i]"""
        self.chromnames2sizes = self.chromsizes
        self.all_chromnames2sizes={}
        self.method = method
        for i in range(len(self.all_chromnames)):
            self.all_chromnames2sizes[self.all_chromnames[i]] = full_chromsizes[self.all_chromnames[i]]
        self.logfile = logfilename

    
    def ExpectedCount(self, coolinfo, ChrList):
        res = {}
        chrom_size = coolinfo.chromsizes
        chrom_cum_size = np.nancumsum(chrom_size)
        chrom_names = chrom_size.keys()
        nbin = coolinfo.binsize
        chridx = [c for c in range(len(chrom_names)) if chrom_names[c] in ChrList]

        for ci in chridx:
            CHROM = chrom_names[ci]
            SIZE = chrom_size[ci]
            if ci == 0:
                r_start = 0
            else:
                r_start = chrom_cum_size[ci - 1] + 1
            r_start = int(np.ceil(r_start / nbin))
            r_end = r_start + int(np.ceil(SIZE / nbin))

            mean_list = []

            for i in range(0,self.windowSize+1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    val = np.nanmean(np.diag(self.unbalLib[r_start:r_end,r_start:r_end], i))
                mean_list.append(val)
            res[CHROM] = mean_list
        return res


    # AG 2024-10-13: Edited
    def getQuantile(self, coolinfo, ChrList, quantile):
        # AG 2024-10-13: CHROMSIZE -> r_end
        def SplitVal(w):
            with np.errstate(divide='ignore', invalid='ignore'):
                w_start = 5000 * w
                w_start += r_start
                w_end = min(5000 * w + 4999, int(np.floor(r_end / nbin)))
                w_end += r_start

                w_start = int(w_start)
                w_end = int(w_end)
                # TODO
                temp = self.unbalLib[w_start:w_end, w_start:w_end][self.unbalLib[w_start:w_end, w_start:w_end] > 0]
                return temp

        res = {}
        # pull chromsizes from coolinfo
        chrom_size = coolinfo.chromsizes
        # cumulatively sum chrom sizes (chrom_cum_size : list)
        chrom_cum_size = np.nancumsum(chrom_size) 
        chrom_names = chrom_size.keys()
        nbin = coolinfo.binsize
        # Index for selected chromosomes
        chridx = [c for c in range(len(chrom_names)) if chrom_names[c] in ChrList]
        chridx.sort()

        # for each chromosome
        for ci in chridx:
            CHROM = chrom_names[ci]
            CHROMSIZE = chrom_size[ci]
            # AG 2024-10-13: Added regions
            REGIONS = self.chromsizes[CHROM]
            
            # if index = 0, start from zero, else go to next item in cumulative sum (start of a new chromosome)
            if ci == 0:
                chr_start = 0
            else:
                chr_start = chrom_cum_size[ci - 1]
            # AG 2024-10-13: only take regions specified in bed
            for start, end in REGIONS:
                # get index holding region start based on bin size
                r_start = np.ceil((chr_start + start) / nbin) + 1
                r_end = np.ceil((chr_start + end) / nbin) + 1
                #hs = np.empty(0)
                # number of windows per region
                N_windows = int(np.ceil((start-end) / nbin / 5000))
                # AG 2024-10-14: Changed n_jobs to run on num cores specified
                W = Parallel(n_jobs=self.core)(delayed(SplitVal)(ww) for ww in tqdm(range(N_windows)))
                #for i in range(len(W)):
                #    hs = np.concatenate((hs, W[i]))
                qt = np.quantile(np.concatenate(W), quantile)
                res[CHROM] = qt
                #del(hs)
        return res

    # AG 2024-10-13: Edited
    def getQuantile_slow(self, coolinfo, ChrList, quantile):
        # AG 2024-10-13: CHROMSIZE -> region_end
        def SplitVal(k,w,r_start,r_end, region_end):
            with np.errstate(divide='ignore', invalid='ignore'):
                w_start = k * w
                w_start += r_start
                w_end = min((k + 1) * w - 1, int(np.floor(region_end / nbin)))
                w_end += r_start
                # TODO
                w_start = int(w_start)
                w_end = int(w_end)
                r_start = int(r_start)
                r_end = int(r_end)
                #print(f"SplitVal: {w_start}, {w_end}, {r_start}, {r_end}")
                temp = self.unbalLib[w_start:w_end, r_start:r_end][self.unbalLib[w_start:w_end, r_start:r_end] > 0]
                return temp

        res = {}
        chrom_size = coolinfo.chromsizes
        chrom_cum_size = np.nancumsum(chrom_size)
        chrom_names = chrom_size.keys()
        nbin = coolinfo.binsize
        # Index for selected chromosomes
        chridx = [c for c in range(len(chrom_names)) if chrom_names[c] in ChrList]
        chridx.sort()
        
        for ci in chridx:
            CHROM = chrom_names[ci]
            #CHROMSIZE = chrom_size[ci]
            #print(self.chromsizes)
            REGIONS = self.chromsizes[CHROM]
            # res[CHROM] = []
            def region_get_quantile(region_start, region_end):
                L = int(np.ceil((region_end - region_start) / nbin))
                #print(f"Processing region {region_start}-{region_end} with {L} bins")
                # TODO
                if ci == 0:
                    r_start = region_start
                else:
                    r_start = region_start + chrom_cum_size[ci - 1]
                

                r_end = region_end + chrom_cum_size[ci]

                r_start = np.ceil(r_start / nbin) + 1
                r_end = np.ceil(r_end / nbin)

                #print("region start", r_start)
                #print("region end", r_end)

                #w = int(np.floor(25000000 / L))

                N_windows_region = int(np.ceil(L / self.windowSize))

                #W = Parallel(n_jobs=self.core)(delayed(SplitVal)(k, self.windowSize, r_start, r_end, region_end) for k in tqdm(range(N_windows_region)))
                W = [SplitVal(k, self.windowSize, r_start, r_end, region_end) for k in range(N_windows_region)]
                # print("W", W)
                hs = np.empty(0)
                for i in range(len(W)):
                    hs = np.concatenate((hs, W[i]))
                qt = np.quantile(hs, quantile)
                del(hs)
                return qt
            
            res[CHROM] = Parallel(n_jobs=self.core)(delayed(region_get_quantile)(REGIONS[i][0], REGIONS[i][1]) for i in tqdm(range(len(REGIONS))))
            # res[CHROM] = [region_get_quantile(REGIONS[i][0], REGIONS[i][1]) for i in tqdm(range(len(REGIONS)))]
            # print(len(res[CHROM]), "length of result")
        return res

    # AG 2024-10-13: Edited
    def getQuantile_original(self, coolinfo, ChrList, quantile):

        res = {}
        chrom_size = coolinfo.chromsizes
        chrom_names = chrom_size.keys()
        # Index for selected chromosomes
        chridx = [c for c in range(len(chrom_names)) if chrom_names[c] in ChrList]

        chridx.sort()
        for ci in chridx:
            CHROM = chrom_names[ci]
            res[CHROM] = []
            REGIONS = self.chromsizes[CHROM]
            for start, end in REGIONS:
                # TODO rectangle
                mat = self.unbalLib.fetch(f"{CHROM}:{start}-{end}")
                qt = np.quantile(mat[mat>0], quantile)
                res[CHROM].append(qt)
                del mat
        return res

    # AG 2024-10-13: Edited to work by-region
    def mpmean(self):
        # TODO: vectorize
        # TODO rectangle
        def calc(n, region_size, framesize):
            with np.errstate(divide='ignore', invalid='ignore'):
                pixels_mean = np.zeros(framesize)
                counts_mean = np.zeros(framesize)
                start = self.resol * n * framesize + 1
                # 1 frame away
                end = min(self.resol * (n + 1) * framesize, region_size)
                # 2 frames away
                end2 = min(self.resol * (n + 2) * framesize, region_size)
                # TODO
                rows = f"{chr}:{start}-{end}"
                cols = f"{chr}:{start}-{end2}"
                cfm = self.unbalLib.fetch(rows, cols)
                cfm_rows, cfm_cols = cfm.shape
                # cfm_max = np.max(cfm)

                for j in range(min(framesize, cfm_cols)):
                    counts = np.diagonal(cfm[:, j:], offset=j)
                    counts = np.nan_to_num(counts, nan=0)
                    pixels_mean[j] += np.sum(counts)
                    counts_mean[j] += len(counts)

                """for i in range(cfm_rows):
                    for j in range(min(framesize, cfm_cols - i)):
                        count = cfm[i, i + j]
                        if np.isnan(count):
                            count = 0
                        pixels_mean[j] += count
                        counts_mean[j] += 1"""
                del cfm
            return pixels_mean, counts_mean

        #meantable = [[] for x in range(len(self.chromnames))]
        # AG 2024-10-14: meantable = [[[region means] for region in self.chromsizes[x]] for x in range(len(self.chromnames))]
        meantable = {}
        for chridx in range(len(self.chromnames)):
            # AG 2024-10-13: Added regions
            chr = self.chromnames[chridx]
            regions = self.chromsizes[chr]
            print('Processing Chromosome: ' + str(chr))
            chrsize = self.chromnames2sizes[chr]
            #means = []
            def region_mean_calc(start, end):
                rowsize = int(np.ceil((end-start) / self.resol))
                framesize = self.windowSize
                nframes = math.ceil(rowsize / framesize)

                result = [calc(n, end-start, framesize) for n in range(nframes)]

                region_means = []
                for i in range(len(result[0][1])):
                    pixelsum = [result[j][0][i] for j in range(nframes)]
                    countsum = [result[j][1][i] for j in range(nframes)]
                    pixelsum = sum(pixelsum)
                    countsum = sum(countsum)
                    meanval = pixelsum/countsum
                    region_means.append(meanval)
                
                return region_means
            with np.errstate(divide='ignore', invalid='ignore'):
                means = Parallel(n_jobs=self.core)(delayed(region_mean_calc)(regions[i][0], regions[i][1]) for i in tqdm(range(len(regions))))
            print(len(means))
            meantable[chr] = means

        return meantable

    # AG 2024-10-14: Edited to work by-region
    def nulldist(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            t_background_start = time.time()
            samplesize = [[end - start for (start, end) in self.chromsizes[chr]] for chr in self.chromnames]
            samplesize = [(sublist / np.sum(sublist))* 1000  for sublist in samplesize]
            samplesize = [np.uint64(sublist) for sublist in samplesize]
            difs = [1000 - np.sum(sublist) for sublist in samplesize]
            print(f"step 3 difs: {difs}")

            notzero = [np.where(sublist != 0)[0] for sublist in samplesize]
            #chromnames2 = [self.all_chromnames[i] for i in notzero[0]]
            regions2_initial = [[self.chromsizes[chr][i] for i in notzero[c]] for c, chr in enumerate(self.chromnames)]
            #for c in range(len(self.chromnames)):
            #    chr = self.chromnames[c]
            #    regions2_initial.append([self.chromsizes[chr][i] for i in notzero[c]])
            for i in range(len(samplesize)):
                samplesize[i][0] += difs[i]

            def available_cols(chr, start, end):
                with np.errstate(divide='ignore', invalid='ignore'):
                    chr = str(chr)
                    #chrsize = self.chromnames2sizes[chr]
                    region_size = end - start
                    itera = min(region_size/self.resol/500, 25)
                    itera = np.uint64(itera)
                    unitsize = np.floor(region_size/self.resol/itera)
                    unitsize = np.uint64(unitsize)
                    poolsum = 0
                    for it in range(itera):
                        test_region_start = np.uint64(unitsize * self.resol * it+1)
                        test_region_end = np.uint64(unitsize * self.resol * (it+1))
                        # test_region_start, test_region_end = min(test_region_start, test_region_end), max(test_region_start, test_region_end)
                        if test_region_start > test_region_end:
                            a = test_region_start
                            test_region_start = test_region_end
                            test_region_end = a
                        position = str(chr) + ":" + str(test_region_start) + "-" + str(test_region_end)
                        mat = self.unbalLib.fetch(position, position)
                        mat = nantozero(mat)
                        #mat = np.round(mat, 1)
                        matsum = np.sum(mat, axis=1)
                        zeroindex = np.where(matsum == 0)
                        poolsum += (len(matsum) - len(zeroindex[0]))

                return poolsum

            # AG 2024-10-14: Edited to work per-region
            print('3.1. Calculating the number of available columns ...')
            samplesize = []
            regions2 = []
            chromnames2 = []
            for chri in range(len(self.chromnames)):
                regions_chr = regions2_initial[chri]
                chr = self.chromnames[chri]             
                n_available_col = Parallel(n_jobs=self.core)(delayed(available_cols)(chr, start, end) for (start,end) in tqdm(regions_chr))

                samplesize_chr = (n_available_col/ np.sum(n_available_col)) * 1000
                samplesize_chr = np.uint64(samplesize)
                dif = 1000 - np.sum(samplesize_chr)
                notzero = np.where(samplesize_chr != 0)[0]
                if notzero.size == 0:
                    continue
                regions2_chr = [regions_chr[i] for i in notzero[0]]
                samplesize_chr[0] += dif
                chromnames2.append(chr)
                samplesize.append(samplesize_chr)
                regions2.append(regions2_chr)

            # TODO: rectangle, refactor
            def main_null_calc(chr, start, end):
                with np.errstate(divide='ignore', invalid='ignore'):
                    background_size = 50000/self.resol
                    background_up = np.floor(background_size / 2)
                    background_down = background_size - background_up
                    background_up = int(background_up)
                    background_down = int(background_down)
                    background_size = int(background_size)
                    chr = str(chr)
                    #c = np.where(chromnames2 == chr)[0]
                    c = chromnames2.index(chr)

                    # Modified in Dec 11 2020
                    # AG 2024-10-14: Work with regions
                    ss = samplesize[c]
                    region_size = end - start
                    itera = min(region_size/self.resol/500, 25)
                    itera = int(itera)
                    unitsize = np.floor(region_size/self.resol/itera)
                    unitsize = int(unitsize)

                    bgleft_up = np.zeros((self.windowSize, 0))
                    bgright_up = np.zeros((self.windowSize, 0))
                    bgleft_down = np.zeros((self.windowSize, 0))
                    bgright_down = np.zeros((self.windowSize, 0))
                    n_pool = []

                    for it in range(itera):
                        sss = int(ss/itera)
                        test_region_start1 = int(unitsize * self.resol * it+1)
                        test_region_start0 = max(test_region_start1 - (self.windowSize*self.resol), 1)
                        test_region_end1 = min(int(unitsize * self.resol * (it+1)), region_size - self.windowSize*self.resol)
                        test_region_end2 = min(int(unitsize * self.resol * (it+1)+(self.windowSize*self.resol)), region_size-1)
                        
                        test_region_start0 = int(test_region_start0)
                        position1 = str(chr) + ":" + str(test_region_start1) + "-" + str(test_region_end1)
                        position2 = str(chr) + ":" + str(test_region_start0) + "-" + str(test_region_end2)
                        mat = self.unbalLib.fetch(position1, position2)
                        mat = nantozero(mat)
                       # mat = np.round(mat, 1)
                        nrow = mat.shape[0]
                        matsum = np.sum(mat, axis=1)
                        zeroindex = np.where(matsum == 0)
                        pool = [x for x in list(range(nrow)) if x not in zeroindex[0].tolist()]
                        pool = [x for x in pool if x > 20 and x < (unitsize - 20)]
                        if it == 0:
                            pool = [x for x in pool if x > 410 and x < mat.shape[1]]
                        n_pool.append(len(pool))
                        if len(pool) == 0:
                            del mat
                        elif len(pool) < sss:
                            randval = self.prng.choices(pool, k=len(pool))
                            tableft_up = np.zeros((self.windowSize, len(pool)))
                            tabcenter_up = np.zeros((self.windowSize, len(pool)))
                            tabright_up = np.zeros((self.windowSize, len(pool)))
                            tableft_down = np.zeros((self.windowSize, len(pool)))
                            tabcenter_down = np.zeros((self.windowSize, len(pool)))
                            tabright_down = np.zeros((self.windowSize, len(pool)))
                            for i in range(len(pool)):
                                x = randval[i]
                                for j in range(0,self.windowSize):
                                    #det = np.random.choice([0,1])
                                    y_down = x + j
                                    y_up = x - j

                                    #if det == 0:
                                    #    y = x + j
                                    #else:
                                    #    y = x - j
                                    if it > 0 :
                                        y_down = y_down + self.windowSize
                                        y_up = y_up + self.windowSize

                                    tableft_up[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_up - background_up):(y_up + background_down)])
                                    tabcenter_up[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_up - background_up):(y_up + background_down)])
                                    tabright_up[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_up - background_up):(y_up + background_down)])

                                    tableft_down[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_down - background_up):(y_down + background_down)])
                                    tabcenter_down[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_down - background_up):(y_down + background_down)])
                                    tabright_down[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_down - background_up):(y_down + background_down)])


                            bgleft_up_temp = np.subtract(tabcenter_up, tableft_up)
                            bgright_up_temp = np.subtract(tabcenter_up, tabright_up)
                            bgleft_up = np.column_stack((bgleft_up, bgleft_up_temp))
                            bgright_up = np.column_stack((bgright_up, bgright_up_temp))
                            bgleft_down_temp = np.subtract(tabcenter_down, tableft_down)
                            bgright_down_temp = np.subtract(tabcenter_down, tabright_down)
                            bgleft_down = np.column_stack((bgleft_down, bgleft_down_temp))
                            bgright_down = np.column_stack((bgright_down, bgright_down_temp))

                            del mat
                        else:
                            randval = self.prng.choices(pool, k=sss)
                            tableft_up = np.zeros((self.windowSize, sss))
                            tabcenter_up = np.zeros((self.windowSize, sss))
                            tabright_up = np.zeros((self.windowSize, sss))
                            tableft_down = np.zeros((self.windowSize, sss))
                            tabcenter_down = np.zeros((self.windowSize, sss))
                            tabright_down = np.zeros((self.windowSize, sss))
                            for i in range(sss):
                                x = randval[i]
                                for j in range(0, self.windowSize):
                                    y_down = x + j
                                    y_up = x - j

                                    if it > 0:
                                        y_up = y_up + self.windowSize
                                        y_down = y_down + self.windowSize
                                    tableft_up[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_up - background_up):(y_up + background_down)])
                                    tabcenter_up[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_up - background_up):(y_up + background_down)])
                                    tabright_up[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_up - background_up):(y_up + background_down)])
                                    tableft_down[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_down - background_up):(y_down + background_down)])
                                    tabcenter_down[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_down - background_up):(y_down + background_down)])
                                    tabright_down[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_down - background_up):(y_down + background_down)])

                            bgleft_up_temp = np.subtract(tabcenter_up, tableft_up)
                            bgright_up_temp = np.subtract(tabcenter_up, tabright_up)
                            bgleft_down_temp = np.subtract(tabcenter_down, tableft_down)
                            bgright_down_temp = np.subtract(tabcenter_down, tabright_down)

                            del mat
                            bgleft_up = np.column_stack((bgleft_up, bgleft_up_temp))
                            bgright_up = np.column_stack((bgright_up, bgright_up_temp))
                            bgleft_down = np.column_stack((bgleft_down, bgleft_down_temp))
                            bgright_down = np.column_stack((bgright_down, bgright_down_temp))

                    depl = int(ss - bgleft_up.shape[1])
                    if depl > 0:
                        rich = np.argmax(n_pool)
                        test_region_start1 = int(unitsize * self.resol * rich+1)
                        test_region_start0 = int(test_region_start1 - (self.windowSize*self.resol))
                        test_region_end1 = int(unitsize * self.resol * (rich+1))
                        test_region_end2 = int(unitsize * self.resol * (rich+1)+(self.windowSize*self.resol))
                        if test_region_end2 > region_size:
                            test_region_end2 = region_size-1
                        if test_region_end1 > region_size - self.windowSize*self.resol:
                            test_region_end1 = region_size - self.windowSize*self.resol
                        if test_region_start0 <= 1:
                            test_region_start0 = 1

                        position1 = str(chr) + ":" + str(test_region_start1) + "-" + str(test_region_end1)
                        position2 = str(chr) + ":" + str(test_region_start0) + "-" + str(test_region_end2)
                        mat = self.unbalLib.fetch(position1, position2)
                        mat = nantozero(mat)
                        #mat = np.round(mat, 1)
                        nrow = mat.shape[0]
                        matsum = np.sum(mat, axis=1)
                        zeroindex = np.where(matsum == 0)
                        pool = [x for x in list(range(nrow)) if x not in zeroindex[0].tolist()]
                        pool = [x for x in pool if x > 20 and x < (unitsize - 20)]
                        randval = self.prng.choices(pool, k=depl)
                        tableft_up = np.zeros((self.windowSize, depl))
                        tabcenter_up = np.zeros((self.windowSize, depl))
                        tabright_up = np.zeros((self.windowSize, depl))
                        tableft_down = np.zeros((self.windowSize, depl))
                        tabcenter_down = np.zeros((self.windowSize, depl))
                        tabright_down = np.zeros((self.windowSize, depl))
                        for i in range(depl):
                            x = randval[i]
                            for j in range(0, self.windowSize):
                                y_down = x + j
                                y_up= x - j
                                #det = np.random.choice([0, 1])

                                #if det == 0:
                                #    y = x + j
                                #else:
                                #    y = x - j
                                if it > 0:
                                    y_up = y_up + self.windowSize
                                    y_down = y_down+ self.windowSize
                                tableft_up[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_up - background_up):(y_up + background_down)])
                                tabcenter_up[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_up - background_up):(y_up + background_down)])
                                tabright_up[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_up - background_up):(y_up + background_down)])
                                tableft_down[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_down - background_up):(y_down + background_down)])
                                tabcenter_down[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_down - background_up):(y_down + background_down)])
                                tabright_down[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_down - background_up):(y_down + background_down)])

                        bgleft_up_temp = np.subtract(tabcenter_up, tableft_up)
                        bgright_up_temp = np.subtract(tabcenter_up, tabright_up)
                        bgleft_down_temp = np.subtract(tabcenter_down, tableft_down)
                        bgright_down_temp = np.subtract(tabcenter_down, tabright_down)

                        del mat
                        bgleft_up = np.column_stack((bgleft_up, bgleft_up_temp))
                        bgright_up = np.column_stack((bgright_up, bgright_up_temp))
                        bgleft_down = np.column_stack((bgleft_down, bgleft_down_temp))
                        bgright_down = np.column_stack((bgright_down, bgright_down_temp))

                return bgleft_up, bgright_up, bgleft_down, bgright_down
            
            # apply parallel.
            # AG 2024-10-14: Edited to work per-region
            print('3.2. Constituting background ...')
            result = []
            bgleft_up = np.zeros((self.windowSize,0))
            bgright_up = np.zeros((self.windowSize,0))
            bgleft_down = np.zeros((self.windowSize,0))
            bgright_down = np.zeros((self.windowSize,0))
            for i in range(len(regions2)):
                chr = chromnames2[i]
                result_chr = Parallel(n_jobs=self.core)(delayed(main_null_calc)(chr, start, end) for (start,end) in tqdm(regions2[i]))
                for i in range(len(result_chr)):
                    if(type(result_chr[i]) == type(None)):
                        continue
                    else:
                        blu,bru,bld,brd = result_chr[i]
                        bgleft_up=np.column_stack((bgleft_up,blu))
                        bgright_up=np.column_stack((bgright_up,bru))
                        bgleft_down=np.column_stack((bgleft_down,bld))
                        bgright_down=np.column_stack((bgright_down,brd))
                result.append(result_chr)

            with open(self.logfile, 'a') as f:
                f.write('Elapsed time for background estimation: ' + str(time.time() - t_background_start) + ' s\n')
            #print('Elapsed time for background estimation: ' + str(np.round((time.time() - t_background_start) / 60, 3)) + ' min')
            return bgleft_up, bgright_up, bgleft_down, bgright_down

    # TODO
    def getMean(self, df, mask='0'):

        if mask != '0':
            mask = mask.split(':')
            mask_chr = mask[0]
            mask_start = int(mask[1].split('-')[0])
            mask_end = int(mask[1].split('-')[1])

        # TODO rectangle
        def iterate_idx(i):
            np.seterr(divide='ignore', invalid='ignore')
            chr = df['chr'].iloc[i]
            x_start_index = int(df['pos1'].iloc[i])
            x_end_index = int(df['pos2'].iloc[i])
            y_start_index = int(df['pos3'].iloc[i])
            y_end_index = int(df['pos4'].iloc[i])
            region1=str(chr)+':'+str(y_start_index)+'-'+str(y_end_index)
            region2=str(chr)+':'+str(x_start_index)+'-'+str(x_end_index)
            center = self.unbalLib.fetch(region1, region2)

            centerm = np.nanmean(center)
            centersum=np.nansum(center)
            return i,centerm,centersum

        # Scoring
        ### Sobel-like operators
        nrows=len(df)
        result = Parallel(n_jobs=self.core)(delayed(iterate_idx)(i) for i in tqdm(range(nrows)))
        #listM = [0 for x in range(0,nrows)]
        #listS = [0 for x in range(0, nrows)]

        listM, listS = zip(*[(result[r][1], result[r][2]) for r in range(len(result))])

        #for r in range(len(result)):
        #    i,MEAN,SUM = result[r]
        #    listM[i] = MEAN
        #    listS[i] = SUM
        return listM,listS
    
    # i is on the x axis
    def _compute_interaction_sum(self, i, window_size, matrix, height):
        height_px = height//self.resol
        start_index = max(0, i - height_px//2)
        end_index = min(matrix.shape[1], i + height_px//2 + 1)
        
        # Exclude the diagonal value by summing the upper and lower parts separately
        #height = matrix.shape[0]
        sum_upper = np.nansum(matrix[start_index:i, i])
        sum_lower = np.nansum(matrix[i:end_index, i])

        strand = '+' if sum_upper > sum_lower else '-'
        return sum_upper + sum_lower, sum_upper, sum_lower, strand

    def calculate_multiscale_lambda (self, chromosome, peak_center_x_coord_start, peak_center_x_coord_end, peak_center_y_coord_start, peak_center_y_coord_end, lambda_window_sizes):
        window_sizes = sorted(np.array(lambda_window_sizes))
        window_sizes_px = window_sizes//self.resol
        # all_chromnames2sizes
        chrLen = self.all_chromnames2sizes[chromosome]
        region_len_bp = int(peak_center_x_coord_end - peak_center_x_coord_start)
        region_len_px = int(peak_center_x_coord_end - peak_center_x_coord_start)//self.resol
        height = int(peak_center_y_coord_end - peak_center_y_coord_start)
        # if peak_center_x_coord_start < window_sizes[-1]//2 or peak_center_x_coord_end > chrLen  - window_sizes[-1]//2:
            
        remaining_len =  max(window_sizes[-1] - region_len_bp,0)
        
        bound1 = max(peak_center_x_coord_start - remaining_len//2, 0)
        bound2 = min(peak_center_x_coord_end + remaining_len//2, chrLen)

        y_bound1 = max(bound1 - height//2, 0)
        y_bound2 = min(bound2 + height//2, chrLen)
        #try:
        #print(chromosome)

        matrix = self.unbalLib.fetch(f"{chromosome}:{int(y_bound1)}-{int(y_bound2)}",f"{chromosome}:{int(bound1)}-{int(bound2)}")
        #plt.imshow(matrix) ; plt.savefig("matrix.png")
        #print("mat shape", matrix.shape)
        #print(window_sizes_px[-1])
        interaction_sums_and_strands = [self._compute_interaction_sum(i, window_sizes_px[-1], matrix, height) for i in range(matrix.shape[1])]
        sum_uppers = np.array([item[1] for item in interaction_sums_and_strands])
        #print("sum uppers len", len(sum_uppers))
        sum_lowers = np.array([item[2] for item in interaction_sums_and_strands])
        #print(sum_lowers)
        peak_center_idx = matrix.shape[1]//2
        peak_start_idx = remaining_len//2//self.resol
        peak_end_idx =  window_sizes[-1]//self.resol - remaining_len//2//self.resol
        
        mean_upper_signals = []
        std_dev_upper_signals = []
        mean_lower_signals = []
        std_dev_lower_signals = []
        for window_size in window_sizes:
            #print(f"window size: {window_size}")
            remaining_len =  window_size//self.resol - (region_len_px % (window_size//self.resol))
            
            start_index = max(peak_start_idx - remaining_len//2, 0)
            end_index = min(peak_end_idx + remaining_len//2, chrLen//self.resol)

            #start_index = peak_start_idx - window_size//2//self.resol
            #end_index = peak_end_idx + window_size//2//self.resol + 1
            #print(sum_uppers[start_index:end_index])
            #print(sum_lowers[start_index:end_index])
            #print(len(sum_lowers[start_index:end_index]))
            # print(start_index, end_index)
            sum_upper = sum_uppers[start_index:end_index]
            sum_lower = sum_lowers[start_index:end_index]
            mean_upper_signals.append(np.mean(sum_upper))
            std_dev_upper_signals.append(np.std(sum_upper))
            mean_lower_signals.append(np.mean(sum_lower))
            std_dev_lower_signals.append(np.std(sum_lower))

        # print("peak start index", peak_start_idx, "peak end index", peak_end_idx)
        sum_center_uppers = sum_uppers[peak_start_idx:peak_end_idx]
        sum_center_lowers = sum_lowers[peak_start_idx:peak_end_idx]
        #print("sum lowers", sum_lowers)
        # print("sum_center_uppers", sum_center_uppers, "sum_center_lowers", sum_center_lowers)

        mean_center_upper = np.mean(sum_center_uppers)
        mean_center_lower = np.mean(sum_center_lowers)

        # print(mean_center_upper, mean_center_lower)
            
        return mean_center_upper, mean_center_lower, mean_upper_signals, mean_lower_signals, window_sizes

    def pvalue(self, df):
        PVAL = []
        dfsize = len(df)
        pval_start_time = time.time()
        # 2024-10-16: changed to fit by-region
        with np.errstate(divide='ignore',invalid='ignore'):
            def calc_pval(i):
                chr = df['chr'].iloc[i]
                chr = str(chr)
                #chrLen = self.all_chromnames2sizes[chr]
                regionStart = df['region_start'].iloc[i]
                regionEnd = df['region_end'].iloc[i]
                pos1 = df['pos1'].iloc[i]
                pos2 = df['pos2'].iloc[i]
                pos3 = df['pos3'].iloc[i]
                pos4 = df['pos4'].iloc[i]

                #print(f"region: {regionStart}-{regionEnd}")
                #print(f"stripe location: {pos1}-{pos2}")
                #print(self.locations.loc[(self.locations["chrom"] == chr) & (self.locations["start"] >= regionStart) & (self.locations["end"] <= regionEnd),])

                #prior_upper = self.locations.loc[(self.locations["chrom"] == chr) & (self.locations["start"] >= regionStart) & (self.locations["end"] <= regionEnd), "thickStart"]
            
                #prior_lower = self.locations.loc[(self.locations["chrom"] == chr) & (self.locations["start"] >= regionStart) & (self.locations["end"] <= regionEnd), "thickEnd"]
                #lambda_prior_upper = np.mean(prior_upper)
                #lambda_prior_lower = np.mean(prior_lower)
                #print("lambda_prior_upper", lambda_prior_upper, "lambda_prior_lower", lambda_prior_lower)

                
                #signal_in_stripe = self.signal.loc[(self.signal["chrom"] == chr) & (self.signal["start"] >= pos1) & (self.signal["end"] <= pos2),]

                mean_center_upper, mean_center_lower, mean_upper_signals, mean_lower_signals, window_sizes = self.calculate_multiscale_lambda(chr, pos1, pos2, pos3, pos4, [10000,100000,1000000])
                
                #print("pvalues")
                #print("region", df['region_idx'].iloc[i])
                #print(poisson.sf(mean_center_upper, max(mean_upper_signals)))
                #print(poisson.sf(mean_center_lower, max(mean_lower_signals)))
                return min(poisson.sf(mean_center_upper, max(mean_upper_signals)), poisson.sf(mean_center_lower, max(mean_lower_signals)))
            
            PVAL = Parallel(n_jobs=self.core)(delayed(calc_pval)(i) for i in tqdm(range(dfsize)))
            print(PVAL)

        with open(self.logfile, 'a') as f:
            f.write('Elapsed time for p-value calculation: ' + str(time.time() - pval_start_time) + ' s\n')
        #print(PVAL) 
        return PVAL
    
    def scoringstripes(self, df, image_output, window_sizes = [10000, 100000, 1000000]):
        score_start_time = time.time()
        Y_SCORE = []
        dfsize = len(df)
        # 2024-10-16: changed to fit by-region
        with np.errstate(divide='ignore',invalid='ignore'):
            def calc_score(i):
                scores = []
                chr = df['chr'].iloc[i]
                chr = str(chr)
                chrLen = self.all_chromnames2sizes[chr]
                regionStart = df['region_start'].iloc[i]
                regionEnd = df['region_end'].iloc[i]
                pos1 = df['pos1'].iloc[i]
                pos2 = df['pos2'].iloc[i]
                pos3 = df['pos3'].iloc[i]
                pos4 = df['pos4'].iloc[i]
                region_len = pos2-pos1
                region_height = pos4-pos3
                #print("len", region_len, "height", region_height)

                for window_size in window_sizes:
                    #print(f"window size: {window_size}")
                    #print(window_size)
                    remaining_len_x =  window_size - region_len
                    start_coord_x = int(float(max(pos1 - remaining_len_x//2, 0)))
                    end_coord_x = int(float(min(pos2 + remaining_len_x//2, chrLen)))

                    remaining_len_y =  window_size - region_height
                    start_coord_y = int(float(max(pos3 - remaining_len_y//2, 0)))
                    end_coord_y = int(float(min(pos4 + remaining_len_y//2, chrLen)))

                    #print("diff x", end_coord_x-start_coord_x)
                    #print("diff y", end_coord_y-start_coord_y)

                    mat = self.unbalLib.fetch(f"{chr}:{start_coord_y}-{end_coord_y}",f"{chr}:{start_coord_x}-{end_coord_x}",)

                    img_outdir = image_output + '/intermediate_images/'
                    #plt.imshow(mat, cmap='hot'), plt.savefig(f"{img_outdir}mat_{chr}_{pos1}_{pos2}_{pos3}_{pos4}_{window_size}.png")

                    start_idx_x = int(max(0, pos1 - start_coord_x)//self.resol)
                    start_idx_y = int(max(0, pos3 - start_coord_y)//self.resol)
                    end_idx_x = int(min(window_size, pos2 - start_coord_x)//self.resol)
                    end_idx_y = int(min(window_size, pos4 - start_coord_y)//self.resol)
                    #print((start_idx_x, end_idx_x))
                    y_mat = mat[:,start_idx_x:end_idx_x]
                    #print(y_mat.shape)
                    #plt.imshow(y_mat, cmap='hot'), plt.savefig(f"{img_outdir}y_mat_{chr}_{pos1}_{pos2}_{pos3}_{pos4}_{window_size}.png")

                    #stripe_mat = mat[start_idx_y:end_idx_y,start_idx_x:end_idx_x]
                    #plt.imshow(stripe_mat, cmap='hot'), plt.savefig(f"{img_outdir}stripe_mat_{chr}_{pos1}_{pos2}_{pos3}_{pos4}_{window_size}.png")
                    scores.append(np.sum(y_mat[start_idx_y:end_idx_y,:])/np.sum(y_mat))
                #print(min(scores))
                return min(scores)
            Y_SCORE = Y_SCORE + Parallel(n_jobs=self.core)(delayed(calc_score)(i) for i in tqdm(range(dfsize)))
        with open(self.logfile, 'a') as f:
            f.write('Elapsed time for scoring stripes: ' + str(time.time() - score_start_time) + ' s\n')
        return Y_SCORE

    # 2024-10-14: Edited to work by-region
    def extract(self, MP, index, perc , bgleft_up, bgright_up, bgleft_down, bgright_down, image_output):
        with np.errstate(divide='ignore', invalid='ignore'):
            # AG 2024-10-15: Added region details
            def search_frame_region(r_idx, r_start, r_end):
                #print(idx)
                region_len_px = int(np.floor((r_end - r_start)/self.resol))
                remaining_len =  self.windowSize - (region_len_px % self.windowSize)

                start_px = r_start//self.resol - remaining_len//2
                end_px = r_end//self.resol + remaining_len//2
                #start = idx * self.windowSize 
                #end = (idx + 1) * self.windowSize 
                #if end >= rowsize:
                #    end = rowsize - 1
                #if idx == 0:
                #    start = 0
                framesize = max(self.windowSize, end_px - start_px + 1)
                start_array = [(start_px + j) * self.resol + 1 for j in range(framesize)]
                end_array = [s + self.resol - 1 for s in start_array]
                #last = end_array[-1]
                if end_array[-1] <= start_array[0]:
                    return 0

                #print(len(start_array), "start array")
                #print(len(end_array), "end array")

                region = chrname + str(":") + str(r_start) + str('-') + str(r_end)
                locus = chrname + str(":") + str(start_array[0]) + str('-') + str(end_array[-1])
                #print("region:", region)
                #print("locus:", locus)
                try:
                    D = self.unbalLib.fetch(locus, region)
                except:
                    return 0
                #print(D.shape)
                D = nantozero(D)

                # Remove rows and columns containing only zero.
                colsum = np.sum(D, axis=0)
                rowsum = np.sum(D, axis=1) # rowsum == colsum
                #print("colsum", colsum)
                #print("rowsum", rowsum)
                col_nonzero_idx = np.where(colsum != 0)[0]  # data type: tuple
                row_nonzero_idx = np.where(rowsum != 0)[0]
                framesize = len(col_nonzero_idx)
                if framesize > 2:
                    D = D[np.ix_(row_nonzero_idx, col_nonzero_idx)]
                    # print("D", D)
                    if np.size(D) == 0:
                        return 0

                    if np.sum(D) == 0:
                        return 0
                    start_array = [start_array[s] for s in col_nonzero_idx]
                    end_array = [end_array[s] for s in col_nonzero_idx]
                    #print("r_idx", r_idx)
                    if self.method == "canny":
                        temp_res = self.StripeSearch_orig(D, 0, start_px, end_px, M[r_idx][index], perc, chrname, framesize, start_array, end_array, image_output, r_idx, r_start, r_end)
                    else:
                        temp_res = self.StripeSearch(D, 0, start_px, end_px, M[r_idx][index], perc, chrname, framesize, start_array, end_array, image_output, r_idx, r_start, r_end, start_array[0], end_array[-1])
                    return(temp_res)
                else:
                    return 0

            # AG 2024-10-15: Added region details
            result = pd.DataFrame(columns=['chr', 'pos1', 'pos2', 'chr2', 'pos3', 'pos4', 'length', 'width', 'total', 'Mean',
                         'maxpixel', 'num', 'start', 'end', 'x', 'y', 'h', 'w', 'medpixel', 'region_idx', 'region_start', 'region_end'])
            # 2024-10-14: Edited to work by-region
            for chridx in range(len(self.chromnames)):
                t_chr_search_start = time.time()
                chrname = self.chromnames[chridx]
                chrsize = self.all_chromnames2sizes[chrname]
                print("Chrsize",chrsize)
                print('Chromosome: ' + chrname + " / Maximum pixel: " + str(round(perc*100,3))+"%")
                print(int(np.ceil(chrsize / self.resol / self.windowSize)))
                M = MP[chrname]
                print("M length", len(M))
                regions = self.chromsizes[chrname]
                #print("number of regions", len(regions))
                #print("regions", regions)
                def main_search(i):
                    region_start, region_end = regions[i]
                    #print("About to call search_frame_region", i, region_start, region_end)
                    results = search_frame_region(i, region_start,  region_end)
                    if type(results) != int:
                        #print(results)
                        return results
                        #result = result.append(results[n])
                chr_results = pd.concat(Parallel(n_jobs=self.core)(delayed(main_search)(i) for i in tqdm(range(len(regions)))))
                result = pd.concat([result, chr_results])
                with open(self.logfile, 'a') as f:
                    f.write(f"Chromosome {chrname} search time: {time.time()-t_chr_search_start}\n")
            #res = self.RemoveRedundant(result, 'size')
            #res = res.reset_index(drop=True)

            # Stripe filtering and scoring
            # res2 = self.scoringstripes(res)
            p = self.pvalue(result)
            #s = self.scoringstripes(res)
            result = result.assign(pvalue=p)
            #res = res.assign(Stripiness=pd.Series(s[5]))

        return result

    # Called in extract()
    # AG 2024-10-15: Added region details

    def StripeSearch(self, submat, num, start, end, M, perc, chr, framesize, start_array, end_array, image_output, region_idx, region_start, region_end, locus_start, locus_end):
        with np.errstate(divide='ignore', invalid='ignore'):
            res_chr = [];
            res_pos1 = [];
            res_pos2 = [];
            res_pos3 = [];
            res_pos4 = [];
            res_length = [];
            res_width = [];
            res_total = [];
            res_Mean = [];
            res_maxpixel = [];
            res_num = [];
            res_start = [];
            res_end = [];
            res_x = [];
            res_y = [];
            res_h = [];
            res_w = [];
            res_medpixel = [];
            region_idx = region_idx;
            region_start = region_start;
            region_end = region_end;
            mp = perc
            #print(submat)
            #print(submat[submat > 0])
            medpixel = np.quantile(a=submat[submat > 0], q=0.5)
            st = start
            en = end
            S = framesize
            
            blue = 255 * (M - submat) / M
            blue[np.where(blue < 0)] = 0
            green = blue
            red = np.ones(blue.shape) * 255
            

            img = cv.merge((red / 255, green / 255, blue / 255))
            img = np.clip(img, a_min=0, a_max=1)
            
            #print(start_array)
            #print(end_array)
        

            with np.errstate(divide='ignore', invalid='ignore'):
                test_column = []
                end_points = []
                start_points = []
                updown = []  # up = 1, down = 2

                im_gray = cv.cvtColor(np.float32(img), cv.COLOR_RGB2GRAY)
                #print("grayscale parts")
                #print(im_gray.shape, "vs", img.shape)
                
                if self.method == 'lococo':
                    corners, gx, gy, R = ImageProcessing.low_complexity_corner_detector(im_gray, kernel_size=6, window_size=6, k=0.001, threshold=0.0001) 
                elif self.method == 'lococo_torch':
                    corners, gx, gy, R = ImageProcessing.low_complexity_corner_detector_torch(im_gray, kernel_size=6, window_size=6, k=0.001, threshold=0.0001) 
                elif self.method == 'harris':
                    corners = ImageProcessing.harris_corner_detector(im_gray, blockSize=5, kSize=5, k=0.001, threshold_multiplier=0.0001)
                else:
                    exit("Invalid method")

                plt.subplot(211),plt.imshow(img),plt.title('original')
                plt.subplot(212), plt.imshow(im_gray,cmap="gray"), plt.title('Gray + corners')
                for c in corners:
                    plt.subplot(212), plt.plot(c[0], c[1], 'bx', markersize=1)
                
                plt.tight_layout()
                img_outdir = image_output + '/intermediate_images/'
                
                os.makedirs(img_outdir, exist_ok=True)
                plt.savefig(f'{img_outdir}/idx{num}_chr{chr}_{start_array[0]}_{end_array[-1]}_minh{self.minH}_maxw{self.maxW}_res{self.resol}_mp{mp}.png')

                plt.clf()

                if 'lococo' in self.method:
                
                    fig, axs = plt.subplots(nrows = 2, ncols=3, figsize=(50, 25))
                    axsflat = axs.flatten()

                    fig.delaxes(axsflat[0])
                    axsflat[1].imshow(im_gray,cmap="gray")
                    axsflat[2].imshow(im_gray,cmap="gray")
                    axsflat[3].imshow(gx,cmap="gray")
                    axsflat[4].imshow(gy,cmap="gray")
                    axsflat[5].imshow(R,cmap="gray")
                
                    for c in corners:
                        for i in range(2,6):
                            axsflat[i].plot(c[0], c[1], marker=MarkerStyle("o",fillstyle="full"), markersize=20, markerfacecolor="red", markeredgecolor="blue", alpha=0.5)

                    fig.suptitle(f"Lococo - chr{chr}:{region_start}-{region_end}, chr{chr}:{start_array[0]}-{end_array[-1]} - region {region_idx}")
                    plt.tight_layout()
                    plt.savefig(f"{img_outdir}/idx{num}_chr{chr}_{start_array[0]}_{end_array[-1]}_minh{self.minH}_maxw{self.maxW}_res{self.resol}_mp{mp}_Lococo.png")

                    plt.clf()
                
                
                corners_arr = np.array([np.array(c) for c in corners])
                #print(corners_arr)
                if corners_arr.size > 0:
                    x_max, y_max = np.max(corners_arr, axis=0)
                    x_min, y_min = np.min(corners_arr, axis=0)
                    
                    h = y_max - y_min 
                    w = x_max - x_min 
                    res_chr.append(chr)
                    res_pos1.append(x_min*self.resol + region_start) 
                    res_pos2.append(x_max*self.resol + region_start)
                    res_pos3.append(y_min*self.resol + locus_start)
                    res_pos4.append(y_max*self.resol + locus_start)
                    res_length.append(h*self.resol)
                    res_width.append(w*self.resol)
                    res_total.append(submat[y_min:y_max, x_min:x_max].sum())
                    res_Mean.append(submat[y_min:y_max, x_min:x_max].sum() / h / w)
                    res_maxpixel.append(str(mp * 100) + '%')
                    res_num.append(num)
                    res_start.append(start)
                    res_end.append(end)
                    res_x.append(x_min)
                    res_y.append(y_min)
                    res_h.append(h)
                    res_w.append(w)
                    res_medpixel.append(medpixel)
        
            result = pd.DataFrame(
                {'chr': res_chr, 'pos1': res_pos1, 'pos2': res_pos2, 'chr2': res_chr, 'pos3': res_pos3, 'pos4': res_pos4,
                 'length': res_length, 'width': res_width, 'total': res_total, 'Mean': res_Mean,
                 'maxpixel': res_maxpixel, 'num': res_num, 'start': res_start, 'end': res_end,
                 'x': res_x, 'y': res_y, 'h': res_h, 'w': res_w, 'medpixel': res_medpixel, 'region_idx': region_idx, 'region_start':region_start, 'region_end':region_end})

            # result = self.RemoveRedundant(result, 'size')

        return result

    def StripeSearch_orig(self, submat, num, start, end, M, perc, chr, framesize, start_array, end_array, image_output, region_idx, region_start, region_end):
        with np.errstate(divide='ignore', invalid='ignore'):
            res_chr = [];
            res_pos1 = [];
            res_pos2 = [];
            res_pos3 = [];
            res_pos4 = [];
            res_length = [];
            res_width = [];
            res_total = [];
            res_Mean = [];
            res_maxpixel = [];
            res_num = [];
            res_start = [];
            res_end = [];
            res_x = [];
            res_y = [];
            res_h = [];
            res_w = [];
            res_medpixel = [];
            region_idx = region_idx;
            region_start = region_start;
            region_end = region_end;
            mp = perc
            #print(submat)
            #print(submat[submat > 0])
            medpixel = np.quantile(a=submat[submat > 0], q=0.5)
            st = start
            en = end
            S = framesize
            
            blue = 255 * (M - submat) / M
            blue[np.where(blue < 0)] = 0
            green = blue
            red = np.ones(blue.shape) * 255
            

            img = cv.merge((red / 255, green / 255, blue / 255))
            img = np.clip(img, a_min=0, a_max=1)
            #plt.subplot(111),plt.imshow(img),plt.title('original'), plt.show()
            #print(start_array)
            #print(end_array)

            plt.subplot(221),plt.imshow(img),plt.title('original')

            for b in np.arange(0.8, 1.01, 0.1):  # b: brightness parameter
                with np.errstate(divide='ignore', invalid='ignore'):
                    test_column = []
                    end_points = []
                    start_points = []
                    updown = []  # up = 1, down = 2
                    adj = ImageProcessing.imBrightness3D(img, In=([0.0, 0.0, 0.0], [1.0, b, b]),Out=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
                    plt.subplot(222),plt.imshow(adj),plt.title(f'Brightened, b={b}')
                    bfilt = int(self.bfilter) # modified 21 10 23

                    # k should be between 0.04 and 0.06
                    # k higher: fewer false corners
                    # k lower: miss more real corners
                   
                    kernel = np.ones((bfilt, bfilt)) / (bfilt*bfilt) # modified 21 10 23
                    blur = cv.filter2D(adj, -1, kernel)
                    blur = np.clip(blur, a_min=0, a_max=1)
                   
                    gray = cv.cvtColor(np.float32(blur), cv.COLOR_RGB2GRAY)
                    
                    edges = feature.canny(gray, sigma=self.canny)
                    plt.subplot(223),plt.imshow(edges, cmap='gray'),plt.title('Canny edge detection')
                    vert = ImageProcessing.verticalLine(edges, L=60, H=120)

                    plt.subplot(224),plt.imshow(vert, cmap='gray'),plt.title('Vertical line detection')
                    plt.suptitle("Vertical line detection: chr" + chr + " region " + str(region_idx) + " start " + str(region_start) + " end " + str(region_end))
                    plt.tight_layout()
                    img_outdir = image_output + '/intermediate_images/'
                    
                    os.makedirs(img_outdir, exist_ok=True)
                    plt.savefig(f'{img_outdir}/idx{num}_chr{chr}_{start_array[0]}_{end_array[-1]}_minh{self.minH}_maxw{self.maxW}_res{self.resol}_mp{mp}_brightness{b}.png')

                    LL = []
                    for c in range(S):
                        # t1 = time.time()
                        line_length, END = ImageProcessing.block(vert, c)
                        # print(time.time()-t1)
                        LL.append(line_length)
                        above = min(c, END)
                        bottom = max(c, END)
                        seq = list(range(above, bottom + 1, 1))

                        if line_length > self.minH and sum(vert[seq, c]) != 0:
                            test_column.append(c)
                            end_points.append(END)
                            start_points.append(c)
                            if END > c:
                                updown.append(2)
                            else:
                                updown.append(1)

                    Pair = []
                    MIN_vec = []
                    MAX_vec = []
                    for ud in [1, 2]:
                        testmat = np.zeros((S, S), dtype=np.uint8)
                        udidx = [i for i in range(len(updown)) if updown[i] == ud]
                        for c in udidx:
                            st = test_column[c]
                            en = end_points[c]
                            if ud == 1:
                                en_temp = st
                                st = en
                                en = en_temp
                            testmat[st:en, test_column[c]] = 1
                        # line refinement
                        for r in range(S):
                            vec = testmat[r, :]
                            K1 = vec[1:S] > vec[0:(S - 1)]
                            K2 = vec[1:S] < vec[0:(S - 1)]
                            st = [i + 1 for i in range(len(K1)) if K1[i]]
                            en = [i for i in range(len(K2)) if K2[i]]
                            if vec[0] == 1:
                                st.insert(0, 0)
                            if vec[S - 1] == 1:
                                en.insert(len(en), S - 1)

                            nLines = len(st)

                            for L in range(nLines):
                                origLine = edges[r, list(range(st[L], en[L] + 1, 1))]
                                SUM = sum(origLine)
                                if SUM > 0:
                                    testmat[r, st[L]:en[L]] = vert[r, st[L]:en[L]]
                                else:
                                    MED = int(np.round(stat.median([st[L] + en[L]]) / 2))
                                    testmat[r, st[L]:en[L]] = 0
                                    testmat[r, MED] = 1
        
                        [_, Y] = np.where(testmat == 1)
                        uniqueCols = list(set(Y))
                        ps = pd.Series(Y)
                        counts = ps.value_counts().sort_index()
                        counts = counts.to_frame(name='length')
                        start_points_ud = [start_points[i] for i in udidx]
                        end_points_ud = [end_points[i] for i in udidx]
                        intersectidx = [i for i in range(len(start_points_ud)) if start_points_ud[i] in uniqueCols]
                        start_points_ud = [start_points_ud[i] for i in intersectidx]
                        end_points_ud = [end_points_ud[i] for i in intersectidx]

                        counts['end_points'] = end_points_ud

                        counts = counts[counts['length'] >= 3]
                        nrow = counts.shape[0]
                        meanX = []
                        Continuous = []
                        isContinue = False
                        Len = []

                        for c in range(nrow - 1):
                            Current = counts.index[c]
                            Next = counts.index[c + 1]

                            if Next - Current == 1 and isContinue:
                                Continuous.append(Next)
                                Len.append(counts.iloc[c + 1]['length'])
                            elif Next - Current == 1 and not isContinue:
                                Continuous = [Current, Next]
                                Len = [counts.iloc[c]['length'], counts.iloc[c + 1]['length']]
                                isContinue = True
                            elif Next - Current != 1 and not isContinue:
                                Continuous = [Current]
                                Len = [counts.iloc[c]['length']]
                                Len = [a / sum(Len) for a in Len]
                                isContinue = False
                                temp = sum([a * b for a, b in zip(Continuous, Len)])
                                meanX.append(np.round(temp))
                            else:
                                Len = [a / sum(Len) for a in Len]
                                temp = sum([a * b for a, b in zip(Continuous, Len)])
                                meanX.append(np.round(temp))
                                Continuous = [Current]
                                Len = [counts.iloc[c]['length']]
                                isContinue = False
                        Len = [a / sum(Len) for a in Len]
                        temp = sum([a * b for a, b in zip(Continuous, Len)])
                        meanX.append(np.round(temp))

                        X = list(set(meanX))
                        X.sort()
                        Xsize = len(X)

                        for c in range(Xsize - 1):
                            n = int(X[c])
                            m = int(X[c + 1])
                            st1 = np.where(testmat[:, n] == 1)[0]
                            en1 = st1.max()
                            st1 = st1.min()
                            st2 = np.where(testmat[:, m] == 1)[0]
                            en2 = st2.max()
                            st2 = st2.min()
                            '''
                            max_width_dist1.append(abs(m - n))
                            if c == 0:
                                max_width_dist2.append(abs(m - n))
                            else:
                                l = int(X[c - 1])
                                minw = min(abs(m - n), abs(n - l))
                                if max_width_dist2[-1] == minw:
                                    continue
                                else:
                                    max_width_dist2.append(minw)
                            '''
                            if abs(m - n) > 1 and abs(m - n) <= self.maxW:
                                if abs(m - n) > 4:
                                    Pair.append((n, m - 2))
                                else:
                                    Pair.append((n , m))
                                [a1, _] = np.where(testmat[:, range(max(0, n - 1), min(n + 2, S), 1)] == 1)
                                MIN1 = a1.min()
                                MAX1 = a1.max()
                                [a2, _] = np.where(testmat[:, range(max(0, m - 1), min(m + 2, S), 1)] == 1)
                                MIN2 = a2.min()
                                MAX2 = a2.max()

                                MIN = min(MIN1, MIN2)
                                MAX = max(MAX1, MAX2)

                                if ud == 1:
                                    if abs(m - n) > 4:
                                        MAX = m - 2 # X[c + 1]
                                    else:
                                        MAX = m
                                else:
                                    MIN = X[c]
                                MIN_vec.append(MIN)
                                MAX_vec.append(MAX)
                    PairSize = len(Pair)

                    for c in range(PairSize):
                        x = Pair[c][0]
                        y = int(MIN_vec[c])
                        w = int(Pair[c][1] - Pair[c][0] + 1)
                        h = int(MAX_vec[c] - MIN_vec[c] + 1)
                        #print("x",x)
                        #print("width",w)
                        res_chr.append(chr)
                        res_pos1.append(start_array[x])
                        res_pos2.append(end_array[x + w - 1])
                        res_pos3.append(start_array[y])
                        res_pos4.append(end_array[y + h - 1])
                        res_length.append(end_array[y + h - 1] - start_array[y] + 1)
                        res_width.append(end_array[x + w - 1] - start_array[x] + 1)
                        res_total.append(submat[y:(y + h), x:(x + w)].sum())
                        res_Mean.append(submat[y:(y + h), x:(x + w)].sum() / h / w)
                        res_maxpixel.append(str(mp * 100) + '%')
                        res_num.append(num)
                        res_start.append(start)
                        res_end.append(end)
                        res_x.append(x)
                        res_y.append(y)
                        res_h.append(h)
                        res_w.append(w)
                        res_medpixel.append(medpixel)
                        """region_idx.append(region_idx)
                        region_start.append(region_start)
                        region_end.append(region_end)"""

            result = pd.DataFrame(
                {'chr': res_chr, 'pos1': res_pos1, 'pos2': res_pos2, 'chr2': res_chr, 'pos3': res_pos3, 'pos4': res_pos4,
                 'length': res_length, 'width': res_width, 'total': res_total, 'Mean': res_Mean,
                 'maxpixel': res_maxpixel, 'num': res_num, 'start': res_start, 'end': res_end,
                 'x': res_x, 'y': res_y, 'h': res_h, 'w': res_w, 'medpixel': res_medpixel, 'region_idx': region_idx, 'region_start':region_start, 'region_end':region_end})

            # result = self.RemoveRedundant(result, 'size')

        return result

    def RemoveRedundant(self, df, by):
        def clean(n):
            delidx=[]
            n_idx = np.where(subdf['num'] == n)[0]
            n2_idx = np.where(subdf['num'] == n + 1)[0]
            n_idx = np.concatenate((n_idx, n2_idx))
            n_idx.sort()
            L = len(n_idx)
            with np.errstate(divide='ignore', invalid='ignore'):
                for i in range(L - 1):
                    for j in range(i + 1, L):
                        ii = c_idx[n_idx][i]
                        jj = c_idx[n_idx][j]

                        A_x_start = list_pos1[ii]
                        A_x_end = list_pos2[ii]
                        A_y_start = list_pos3[ii]
                        A_y_end = list_pos4[ii]

                        B_x_start = list_pos1[jj]
                        B_x_end = list_pos2[jj]
                        B_y_start = list_pos3[jj]
                        B_y_end = list_pos4[jj]

                        int_x = max(min(A_x_end, B_x_end) + 1 - max(A_x_start, B_x_start),0)
                        int_y = max(min(A_y_end, B_y_end) + 1 - max(A_y_start, B_y_start),0)

                        if min(A_x_end - A_x_start, B_x_end - B_x_start) <= 0:
                            s_x = 0
                        else:
                            s_x = int_x / min(A_x_end - A_x_start, B_x_end - B_x_start)
                        if min(A_y_end - A_y_start, B_y_end - B_y_start) <= 0:
                            s_y = 0
                        else:
                            s_y = int_y / min(A_y_end - A_y_start, B_y_end - B_y_start)

                        # 2024-11-04: modified to remove fewer stripes
                        if s_x > 0.2 and s_y > max(0.2, 5/min(A_y_end - A_y_start, B_y_end - B_y_start)):
                            if by == 'size':
                                if list_h[ii] / list_w[ii] <= list_h[jj] / list_w[jj]: # 2021.06.23 modified
                                    delidx.append(ii)
                                else:
                                    delidx.append(jj)
                            elif by == 'score':
                                if list_stri[ii] <= list_stri[jj]:
                                    delidx.append(ii)
                                else:
                                    delidx.append(jj)
                            else:
                                if list_pval[ii] > list_pval[jj]:
                                    delidx.append(ii)
                                else:
                                    delidx.append(jj)
            return delidx

        if by not in {'size', 'score', 'pvalue'}:
            raise ValueError('"by" should be one of "size", "pvalue" and "score"')

        df_size = df.shape
        row_size = df_size[0]
        if row_size == 0:
            return df
        else:
            delobj = [True for i in range(row_size)]
            list_chr = df['chr']
            list_pos1 = df['pos1'].tolist()
            list_pos2 = df['pos2'].tolist()
            list_pos3 = df['pos3'].tolist()
            list_pos4 = df['pos4'].tolist()
            list_h = df['h'].tolist()
            list_w = df['w'].tolist()
            if by == 'score':
                list_stri = df['Stripiness'].tolist()
            if by == 'pvalue':
                list_pval = df['pvalue'].tolist()
            unique_chr = list(set(list_chr))

            for c in unique_chr:
                c_idx = np.where(list_chr == c)[0]
                subdf = df.iloc[c_idx]
                unique_num = list(set(subdf['num']))
                unique_num.sort()
                res = Parallel(n_jobs=self.core)(delayed(clean)(n) for n in unique_num)
                for i in range(len(res)):
                    for j in res[i]:
                        delobj[j] = False

        idx = [a for a in range(row_size) if delobj[a]]
        result = df.iloc[idx]
        return result

    def selectColumn(self, df):
        list_chr = df['chr']
        list_pos1 = df['pos1'].tolist()
        list_pos2 = df['pos2'].tolist()
        list_pos3 = df['pos3'].tolist()
        list_pos4 = df['pos4'].tolist()

        if str(list_chr[0])[0] != 'c' and self.chromnames[0][0] == 'c':
            list_chr = ['chr'+str(x) for x in list_chr]
        MAX_POS=[]
        nrow = df.shape[0]
        for i in range(nrow):
            chr = list_chr[i]
            pos1 = list_pos1[i]
            pos2 = list_pos2[i]
            pos3 = list_pos3[i]
            pos4 = list_pos4[i]

            x_str = str(chr)+':'+str(pos1)+'-'+str(pos2)
            y_str = str(chr)+':'+str(pos3)+'-'+str(pos4)
            mat = self.unbalLib.fetch(x_str,y_str)
            average = np.mean(mat,axis=1)
            which_max = np.argmax(average)
            max_pos = pos1 + self.resol * which_max
            MAX_POS.append(max_pos)


        return MAX_POS


def nantozero(nparray):
    where_are_nans = np.isnan(nparray)
    nparray[where_are_nans] = 0
    return nparray


def extract_from_matrix(matrix, x_start, x_end, y_start, y_end, mask_x_start = 0, mask_x_end = 0):
    x = list(range(x_start, x_end))
    x_mask = list(range(mask_x_start, mask_x_end))
    x = [i for i in x if i not in x_mask]
    xlen = len(x)
    y = list(range(y_start, y_end))
    y = [i for i in y if i not in x_mask]
    ylen = len(y)
    result = np.empty((ylen, xlen), dtype=float)
    for i in range(ylen):
        for j in range(xlen):
            result[i][j] = matrix[y[i]][x[j]]


    return result
