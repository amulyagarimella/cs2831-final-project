import cooler
import getStripe
import os
import shutil
import errno
import pandas as pd
import numpy as np
import warnings
import time
import sys
import pybedtools
import re

def makeOutDir(outdir, delete = True):
    last = outdir[-1]
    if last != '/':
        outdir += '/'
    if os.path.exists(outdir):
        if delete:
            print('All directories and files in %s will be deleted.' % (outdir))
        for filename in os.listdir(outdir):
            file_path = os.path.join(outdir, filename)
            if delete:
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s with the reason: %s' % (file_path, e))
    else:
        try:
            os.makedirs(outdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def addlog(cool, out, bedfilename, signalfilename, region, norm, chrom, canny, minL, maxW, maxpixel, numcores, pvalue, mask, bfilter, seed,windowSize, expansionLength_px, method):
    if out[-1] != '/':
        out += '/'
    outfile=open(out + "detector.log",'w')
    outfile.write('cool: ' + cool + '\n')
    outfile.write('bed: ' + bedfilename + '\n')
    outfile.write('signal: ' + signalfilename + '\n')
    outfile.write('region: ' + region + '\n')
    outfile.write('out: ' + out + '\n')
    outfile.write('norm: ' + norm + '\n')
    outfile.write('chrom: ' + str(chrom) + '\n')
    outfile.write('canny: ' + str(canny) + '\n')
    outfile.write('minL: ' + str(minL) + '\n')            
    outfile.write('maxW: ' + str(maxW) + '\n')                
    outfile.write('maxpixel: ' + str(maxpixel) + '\n')
    outfile.write('num_cores: ' + str(numcores) + '\n')
    outfile.write('pvalue: ' + str(pvalue) + '\n')
    outfile.write('mask: ' + str(mask) + '\n')
    outfile.write('blur filter: ' + str(bfilter) + '\n')
    outfile.write('seed: ' + str(seed) + '\n')
    outfile.write('windowSize: ' + str(windowSize) + '\n')
    outfile.write('expansionLength_px: ' + str(expansionLength_px) + '\n')
    outfile.write('method: ' + str(method) + '\n')
    outfile.write('detector started at ' + time.strftime('%c') + '\n')
    outfile.close()
    return out + "detector.log"
    
    
def compute(cool, out, bedfilename, signalfilename, region, norm, chrom, canny, minL, maxW, maxpixel, numcores, pvalue, mask, slow, bfilter, seed, windowSize, expansionLength_px, method):
    np.seterr(divide='ignore', invalid='ignore')
    t_start = time.time()
    if out[-1] != '/':
        out += '/'
    makeOutDir(out)
    logfilename = addlog(cool, out, bedfilename, signalfilename, region, norm, chrom, canny, minL, maxW, maxpixel, numcores, pvalue, mask, bfilter, seed, windowSize, expansionLength_px, method)
    maxpixel = maxpixel.split(',')
    maxpixel = list(map(float, maxpixel))
    minH = minL
    core = numcores
    pcut = pvalue
    #cool, out, norm, chroms, canny, minH, maxW, maxpixel, core, pcut = argumentParser()
    print('Result will be stored in %s' % (out))

    Lib = cooler.Cooler(cool)
    signal = pd.read_csv(signalfilename, sep='\t')
    resol = Lib.binsize
    PossibleNorm = Lib.bins().columns
    chr_in_names = ""
    all_chromnames = Lib.chromnames
    if 'chr' in all_chromnames[0]:
        chr_in_names = "chr"
    if norm == 'None':
        norm = False
    elif norm == 'weight':
        norm = True
    elif norm not in PossibleNorm:
        print('Possible normalization methods are:')
        print('None')
        for n in range(3,len(PossibleNorm)):
            print(PossibleNorm[n])
        print("Invalid normalization method. Normalization method is forced to None")
        norm = False

    # AG 2024-10-13
    """
    New behavior: 
    - if bedfilename is provided, use it to get all_chromsizes
    - else, full chromosomes for all_chromsizes
    """

    bed = pybedtools.BedTool(bedfilename)

    chromsizes_dict = Lib.chromsizes.to_dict()
    chromsizes_dict = {k:(1,v) for k, v in chromsizes_dict.items()}
    # we still need at least one window around each prior
    expansionLength_px = max(expansionLength_px, 1)
    expansionLength = int(expansionLength_px*resol)
    print(f"Expanding priors by {expansionLength} bp on either side")
    print(chromsizes_dict)
    bed = bed.slop(b=expansionLength, g=chromsizes_dict).sort()

    #bed = bed.merge()
    if region is not None:
        list_region = re.split('\W+', region)
        if len(list_region) == 3 and "chr" in list_region[0]:
            interval_string = '\t'.join(pybedtools.create_interval_from_list(list_region))
            interval = pybedtools.BedTool(interval_string, from_string=True)
            bed = bed.intersect(interval)
        else:
            print("Invalid region format. Region is ignored.")
    bed = bed.to_dataframe()
    if len(bed) == 0:
        sys.exit("Exit: No valid regions to analyze.")
    # data structure for chromosome
    bed_chroms = np.unique(bed.chrom) 

    # get chrom sizes
    all_chromsizes = {f"{chr_in_names}{str(chr).lstrip("chr")}": list(zip(bed[bed.chrom==chr]["start"], bed[bed.chrom==chr]["end"])) for chr in bed_chroms}
    all_chromnames = list(all_chromsizes.keys())

    if len(all_chromnames) == 0:
        sys.exit("Exit: All chromosomes are shorter than 50kb.")
    
    warnflag = False

    # AG 2024-10-13: If chroms and bedfile specified, take the intersection -- i.e. only consider regions from bedfile of chromosomes specified in chroms
    chroms = chrom.split(',')
    # AG 2024-10-13: Adapt specified chroms to match format of chromosomes in cooler file
    chroms = [f"{chr_in_names}{str(chr).lstrip("chr")}" for chr in chroms]
    print("chroms 0", chroms[0])
    if chroms[0] != f'{chr_in_names}all':
        print(f"chroms is not all")
        chroms_filtered = []
        for item in chroms:
            if item in all_chromnames:
                chroms_filtered.append(item)
            else:
                warnings.warn('\nThere is no chromosomes called ' + str(item) + ' in the provided .cool file or it is shorter than 50kb.')
                warnflag = True
        if warnflag:
            warnings.warn('\nThe possible chromosomes are: '+ ', '.join(all_chromnames))
        chromnames = chroms_filtered
        # AG 2024-10-13: Filter chromsizes to match filtered chroms
        chromsizes = {k:all_chromsizes[k] for k in chroms_filtered if k in all_chromsizes} 
    else:
        chromnames = all_chromnames
        chromsizes = all_chromsizes

    unbalLib = Lib.matrix(balance=norm)

    # AG 24-10-12 add bed file
    full_chromsizes =  Lib.chromsizes
    # AG 2024-10-13: Edited object to work by-region
    obj = getStripe.getStripe(unbalLib, bed, signal, resol, minH, maxW, canny, all_chromnames, chromnames, all_chromsizes, chromsizes, core, bfilter, seed, windowSize, full_chromsizes, method, logfilename)
    #print('1. Maximum pixel value calculation ...')
    if slow:
        print("1.1 Slowly estimating Maximum pixel values...")
        # AG 2024-10-13: Edited to work by-region
        MP = getStripe.getStripe.getQuantile_slow(obj, Lib, chromnames, maxpixel)
    else:
        # AG 2024-10-13: Edited to work by-region
        MP = getStripe.getStripe.getQuantile_original(obj, Lib, chromnames, maxpixel)
    print('2. Expected value calculation ...')
    # AG 2024-10-14: Edited to work by-region
    EV = getStripe.getStripe.mpmean(obj)
    print('3. Background distribution estimation ...')
    # AG 2024-10-14: Edited to work by-region
    bgleft_up, bgright_up, bgleft_down, bgright_down = getStripe.getStripe.nulldist(obj)
    print('4. Finding candidate stripes from each chromosome ...')
    # AG 2024-10-15: added region details
    result_table = pd.DataFrame(columns=['chr', 'pos1', 'pos2', 'chr2', 'pos3', 'pos4', 'length', 'width', 'total', 'Mean',
                                   'maxpixel', 'num', 'start', 'end', 'x', 'y', 'h', 'w', 'medpixel','region_idx', 'region_start', 'region_end', 'pvalue'])
    for i in range(len(maxpixel)): 
        perc = maxpixel[i]
        # AG 2024-10-14: Edited to work by-region
        result = obj.extract(MP, i, perc, bgleft_up, bgright_up, bgleft_down, bgright_down, out)
        result_table = pd.concat([result_table, result]) 

    res1 = out + 'result_unfiltered.tsv'


    print('5. Stripiness calculation ...')
    s = obj.scoringstripes(result_table, out)
    result_table.insert(result_table.shape[1],'Stripiness',s,True)


    result_table.to_csv(res1,sep="\t",header=True,index=False)
    result_table_removeredundant = getStripe.getStripe.RemoveRedundant(obj, df=result_table, by='pvalue')
    
    res1_rr = out + 'result_unfiltered_removeredundant.tsv'

    
    result_table_removeredundant.to_csv(res1_rr,sep="\t",header=True,index=False)
    with open(logfilename, 'a') as f:
        f.write('\n' + str(round((time.time()-t_start)/60,3)) + 'min taken.')
        f.write('\n' + str((time.time()-t_start)) + 'sec taken.')
    print('Check the result stored in %s' % (out))
    return 0

def score_only(result_table, cool, out, bedfilename, signalfilename, region, norm, chrom, canny, minL, maxW, maxpixel, numcores, pvalue, mask, slow, bfilter, seed, windowSize, expansionLength_px, method):
    np.seterr(divide='ignore', invalid='ignore')
    t_start = time.time()
    if out[-1] != '/':
        out += '/'

    maxpixel = maxpixel.split(',')
    maxpixel = list(map(float, maxpixel))
    minH = minL
    core = numcores
    pcut = pvalue
    print('Result will be stored in %s' % (out))

    Lib = cooler.Cooler(cool)
    signal = pd.read_csv(signalfilename, sep='\t')
    resol = Lib.binsize
    PossibleNorm = Lib.bins().columns
    chr_in_names = ""
    all_chromnames = Lib.chromnames
    if 'chr' in all_chromnames[0]:
        chr_in_names = "chr"
    if norm == 'None':
        norm = False
    elif norm == 'weight':
        norm = True
    elif norm not in PossibleNorm:
        print('Possible normalization methods are:')
        print('None')
        for n in range(3,len(PossibleNorm)):
            print(PossibleNorm[n])
        print("Invalid normalization method. Normalization method is forced to None")
        norm = False

    bed = pybedtools.BedTool(bedfilename)

    chromsizes_dict = Lib.chromsizes.to_dict()
    chromsizes_dict = {k:(1,v) for k, v in chromsizes_dict.items()}
    # we still need at least one window around each prior
    expansionLength_px = max(expansionLength_px, 1)
    expansionLength = int(expansionLength_px*resol)
    print(f"Expanding priors by {expansionLength} bp on either side")
    print(chromsizes_dict)
    bed = bed.slop(b=expansionLength, g=chromsizes_dict).sort()

    if region is not None:
        list_region = re.split('\W+', region)
        if len(list_region) == 3 and "chr" in list_region[0]:
            interval_string = '\t'.join(pybedtools.create_interval_from_list(list_region))
            interval = pybedtools.BedTool(interval_string, from_string=True)
            bed = bed.intersect(interval)
        else:
            print("Invalid region format. Region is ignored.")
    bed = bed.to_dataframe()
    if len(bed) == 0:
        sys.exit("Exit: No valid regions to analyze.")
    # data structure for chromosome
    bed_chroms = np.unique(bed.chrom) 

    # get chrom sizes
    all_chromsizes = {f"{chr_in_names}{str(chr).lstrip("chr")}": list(zip(bed[bed.chrom==chr]["start"], bed[bed.chrom==chr]["end"])) for chr in bed_chroms}
    all_chromnames = list(all_chromsizes.keys())

    if len(all_chromnames) == 0:
        sys.exit("Exit: All chromosomes are shorter than 50kb.")
    
    warnflag = False

    # AG 2024-10-13: If chroms and bedfile specified, take the intersection -- i.e. only consider regions from bedfile of chromosomes specified in chroms
    chroms = chrom.split(',')
    # AG 2024-10-13: Adapt specified chroms to match format of chromosomes in cooler file
    chroms = [f"{chr_in_names}{str(chr).lstrip("chr")}" for chr in chroms]
    print("chroms 0", chroms[0])
    if chroms[0] != f'{chr_in_names}all':
        print(f"chroms is not all")
        chroms_filtered = []
        for item in chroms:
            if item in all_chromnames:
                chroms_filtered.append(item)
            else:
                warnings.warn('\nThere is no chromosomes called ' + str(item) + ' in the provided .cool file or it is shorter than 50kb.')
                warnflag = True
        if warnflag:
            warnings.warn('\nThe possible chromosomes are: '+ ', '.join(all_chromnames))
        chromnames = chroms_filtered
        # AG 2024-10-13: Filter chromsizes to match filtered chroms
        chromsizes = {k:all_chromsizes[k] for k in chroms_filtered if k in all_chromsizes} 
    else:
        chromnames = all_chromnames
        chromsizes = all_chromsizes

    unbalLib = Lib.matrix(balance=norm)

    # AG 24-10-12 add bed file
    full_chromsizes =  Lib.chromsizes
    # AG 2024-10-13: Edited object to work by-region
    logfilename = out + "/scoring.log"
    with open(logfilename, 'w') as f:
        f.write("Scoring only started at " + time.strftime('%c') + "\n")
        
    obj = getStripe.getStripe(unbalLib, bed, signal, resol, minH, maxW, canny, all_chromnames, chromnames, all_chromsizes, chromsizes, core, bfilter, seed, windowSize, full_chromsizes, method, logfilename)
    
    res1 = out + 'result_unfiltered_scoring.tsv'

    print('Stripiness calculation ...')
    with np.errstate(divide='ignore', invalid='ignore'):
        s = obj.scoringstripes(result_table, out)

    result_table.insert(result_table.shape[1],'Stripiness',s,True)
 
    result_table.to_csv(res1,sep="\t",header=True,index=False)
    result_table_removeredundant = getStripe.getStripe.RemoveRedundant(obj, df=result_table, by='pvalue')
    
    res1_rr = out + 'result_unfiltered_removeredundant_scoring.tsv'
    #res2 = out + 'result_filtered.tsv'
    
    result_table_removeredundant.to_csv(res1_rr,sep="\t",header=True,index=False)
    #res_filter.to_csv(res2,sep="\t",header=True,index=False)
    with open(logfilename, 'a') as f:
        f.write('\n' + str(round((time.time()-t_start)/60,3)) + 'min taken.')
        f.write('\n' + str((time.time()-t_start)) + 'sec taken.')
    print('Check the result stored in %s' % (out))
    return 0

#if __name__ == "__main__":
#    main()

