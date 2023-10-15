import numpy as np
import matplotlib.pyplot as plt

#===========================Data Pre-Processing===========================#

#there's 321 features in the dataset
def load_data(x_dataset_path, y_dataset_path, max_rows=None, usecols=None):
    tx = np.genfromtxt(x_dataset_path, delimiter=",", skip_header=1, max_rows=max_rows, usecols=usecols)
    y = np.genfromtxt(y_dataset_path, delimiter=",", skip_header=1, max_rows=max_rows)

    return tx, y

def normalize(x, c):
    """
    A function to normalize an array, returning the data with 0 mean and 1 std_dev along each column
   
    Args:
        x : A (N, d) shaped array containing heterogeneous data
        c: A (d,) shaped boolean array indicating which variables are categorical

    Returns:
        A (N, d) shaped array (like x), with mean 0 and std_dev 1 along each column
    """
    scalar = ~c
    means = np.mean(x[:, scalar], axis=0) # means has shape (d_scalar), where d_scalar is the #scalar features
    std_dev = np.std(x[:, scalar], axis=0)
    x[:, scalar] = (x[:, scalar] - means / std_dev)
    return x

def rows_with_all_features(x):
    missing_elems = np.isnan(x)
    return np.logical_not(missing_elems.any(axis=1))

def remove_rows_with_missing_features(x, y):

    #lines_before = x.shape[0]
    full_rows = rows_with_all_features(x)
    x = x[full_rows]
    y = y[full_rows]
    #print(f"before/after for N : ({lines_before}, {x.shape[0]})")

    return x,y

def replace_missing_features_with_mean(x, c):
    scalar = ~c
    means = np.nanmean(x[:, scalar], axis=0) # means has shape (d_scalar)
    x[:, scalar] = np.nan_to_num(x[:, scalar], nan=means)
    return x


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N, d), N is the number of samples and d the number of features
        degree: integer.

    Returns:
        poly: numpy array of shape (N, d'), where d' is the total number of polynomial features.
        for example, with d=2, degree = 2, we have 1, f1, f2, f1 * f2, f1², f2², so 6 features
    
    NOTE : We need to find a general way of finding all the possible combinations for all degree and d.
    NOTE : this function should have the advantage of handling the 1 column vector by setting degree=1
    """


    # we want to build an N * d array [[0,...,d], ..., [0,...,d]] to broadcast power it with x
    #exponents = np.arange(0, degree + 1)

    #return x.reshape((N, 1)) ** exponents.reshape((1, degree + 1))

    #easy first step : 1 column, then all xi's, then all xi²'s
    N = x.shape[0]
    d = x.shape[1]
    ones = np.ones((N,1))
    squares = x ** 2
    to_concat = [ones]
    for i in range(1, degree+1):
        to_concat.append(x ** i)
    exp_x = np.concatenate(to_concat, axis=1)
    print(f"expanded x: {exp_x}")
    return exp_x

def feature_specific_processing(x):
    # 27, 28: replace 88 with 0, ignore higher than 30
    print(f"x before = {x}")
    x[:,0:2][x[:,0:2]==88] = 0
    x[:,0:2][x[:,0:2] > 30] = np.nan
    print(f"x after = {x}")
    return x

def one_hot_encoding(x, c):
    """
        Args:
            x: Numpy array of shape (N, d) with some categorical features, can contain NaN values
            c: Numpy array of shape (d,) with boolean values encoding which feature is categorical
        Returns:
            A numpy array of shape (N, d'), where each of the categorical feature was converted to one-hot encoding
    """

    """ how to do more dimensions?
    Say x = [[1, 2], [2, 0]]
    We want [ [0 1 0 0 0 1] , [0 0 1 1 0 0] ]
    
    """
    # we need a cumulative sum thing : cat_counts=[2, 3, 3, 2] -> [0, 2, 5, 8]
    # with this we could do z[cat_x + cum_cat_counts] = 1
    # here cat_x + cum_cat_counts = [[1, 2], [2, 0]] + [0, 3] = [[1, 5], [2, 3]]
    # and z[above] = [[0 1 0 0 0 1], [0 0 1 1 0 0]] -> CORRECT!
    if not c.any():
        return x

    N = x.shape[0]
    cat_x = x[:, c] # x with only categorical features
    cat_x = np.nan_to_num(cat_x, nan=0)
    cat_x = cat_x.astype('int32')
    cat_counts = np.nanmax(cat_x, axis=0)+1 # cat_counts is a (d,) shaped array with the max value for each feature
    d_prime = np.sum(cat_counts, dtype=np.int32)
    print(f"N = {N}, d_prime = {d_prime}")
    z = np.zeros((N, d_prime))
    # we don't want to substract the first, we want to slide everything : [1, 3, 2] -> [1, 4, 6] -> [0, 1, 4]
    cum_cat_counts = np.roll(np.cumsum(cat_counts), 1)
    cum_cat_counts[0] = 0
    indexes = cat_x + cum_cat_counts
    z[np.arange(N), indexes.T] = 1

    return np.concatenate((z, x[:, ~c]), axis=1)


#==========================Plotting==========================#
def scatter_plot():
    ...

def line_plot():
    ...

def line_and_scatter_plot(y, tx, preds):
    plt.scatter(tx[:, 1], y, c='r')
    plt.scatter(tx[:, 1], preds, c='b')
    #plt.plot(tx[:, 1], preds)
    plt.show()


#====================Generate Random Data====================#
def generate_linear_data_with_gaussian_noise(N, d) :

    transform = np.random.rand((d)) * 2 - 1 # linear function mapping a d long feature vector on a number
    #print(f"transform = {transform}")

    X_f = 10.0
    X = np.random.normal(size=(N, d)) * X_f

    noise_f = 6
    noise = np.random.normal(size=N) * noise_f * np.linalg.norm(transform)

    y = transform @ X.T + noise # labels

    return y, X


#========================Random Stuff========================#
def list_features():
    s = "_STATE,FMONTH,IDATE,IMONTH,IDAY,IYEAR,DISPCODE,SEQNO,_PSU,CTELENUM,PVTRESD1,COLGHOUS,STATERES,CELLFON3,LADULT,NUMADULT,NUMMEN,NUMWOMEN,CTELNUM1,CELLFON2,CADULT,PVTRESD2,CCLGHOUS,CSTATE,LANDLINE,HHADULT,GENHLTH,PHYSHLTH,MENTHLTH,POORHLTH,HLTHPLN1,PERSDOC2,MEDCOST,CHECKUP1,BPHIGH4,BPMEDS,BLOODCHO,CHOLCHK,TOLDHI2,CVDSTRK3,ASTHMA3,ASTHNOW,CHCSCNCR,CHCOCNCR,CHCCOPD1,HAVARTH3,ADDEPEV2,CHCKIDNY,DIABETE3,DIABAGE2,SEX,MARITAL,EDUCA,RENTHOM1,NUMHHOL2,NUMPHON2,CPDEMO1,VETERAN3,EMPLOY1,CHILDREN,INCOME2,INTERNET,WEIGHT2,HEIGHT3,PREGNANT,QLACTLM2,USEEQUIP,BLIND,DECIDE,DIFFWALK,DIFFDRES,DIFFALON,SMOKE100,SMOKDAY2,STOPSMK2,LASTSMK2,USENOW3,ALCDAY5,AVEDRNK2,DRNK3GE5,MAXDRNKS,FRUITJU1,FRUIT1,FVBEANS,FVGREEN,FVORANG,VEGETAB1,EXERANY2,EXRACT11,EXEROFT1,EXERHMM1,EXRACT21,EXEROFT2,EXERHMM2,STRENGTH,LMTJOIN3,ARTHDIS2,ARTHSOCL,JOINPAIN,SEATBELT,FLUSHOT6,FLSHTMY2,IMFVPLAC,PNEUVAC3,HIVTST6,HIVTSTD3,WHRTST10,PDIABTST,PREDIAB1,INSULIN,BLDSUGAR,FEETCHK2,DOCTDIAB,CHKHEMO3,FEETCHK,EYEEXAM,DIABEYE,DIABEDU,CAREGIV1,CRGVREL1,CRGVLNG1,CRGVHRS1,CRGVPRB1,CRGVPERS,CRGVHOUS,CRGVMST2,CRGVEXPT,VIDFCLT2,VIREDIF3,VIPRFVS2,VINOCRE2,VIEYEXM2,VIINSUR2,VICTRCT4,VIGLUMA2,VIMACDG2,CIMEMLOS,CDHOUSE,CDASSIST,CDHELP,CDSOCIAL,CDDISCUS,WTCHSALT,LONGWTCH,DRADVISE,ASTHMAGE,ASATTACK,ASERVIST,ASDRVIST,ASRCHKUP,ASACTLIM,ASYMPTOM,ASNOSLEP,ASTHMED3,ASINHALR,HAREHAB1,STREHAB1,CVDASPRN,ASPUNSAF,RLIVPAIN,RDUCHART,RDUCSTRK,ARTTODAY,ARTHWGT,ARTHEXER,ARTHEDU,TETANUS,HPVADVC2,HPVADSHT,SHINGLE2,HADMAM,HOWLONG,HADPAP2,LASTPAP2,HPVTEST,HPLSTTST,HADHYST2,PROFEXAM,LENGEXAM,BLDSTOOL,LSTBLDS3,HADSIGM3,HADSGCO1,LASTSIG3,PCPSAAD2,PCPSADI1,PCPSARE1,PSATEST1,PSATIME,PCPSARS1,PCPSADE1,PCDMDECN,SCNTMNY1,SCNTMEL1,SCNTPAID,SCNTWRK1,SCNTLPAD,SCNTLWK1,SXORIENT,TRNSGNDR,RCSGENDR,RCSRLTN2,CASTHDX2,CASTHNO2,EMTSUPRT,LSATISFY,ADPLEASR,ADDOWN,ADSLEEP,ADENERGY,ADEAT1,ADFAIL,ADTHINK,ADMOVE,MISTMNT,ADANXEV,QSTVER,QSTLANG,MSCODE,_STSTR,_STRWT,_RAWRAKE,_WT2RAKE,_CHISPNC,_CRACE1,_CPRACE,_CLLCPWT,_DUALUSE,_DUALCOR,_LLCPWT,_RFHLTH,_HCVU651,_RFHYPE5,_CHOLCHK,_RFCHOL,_LTASTH1,_CASTHM1,_ASTHMS1,_DRDXAR1,_PRACE1,_MRACE1,_HISPANC,_RACE,_RACEG21,_RACEGR3,_RACE_G1,_AGEG5YR,_AGE65YR,_AGE80,_AGE_G,HTIN4,HTM4,WTKG3,_BMI5,_BMI5CAT,_RFBMI5,_CHLDCNT,_EDUCAG,_INCOMG,_SMOKER3,_RFSMOK3,DRNKANY5,DROCDY3_,_RFBING5,_DRNKWEK,_RFDRHV5,FTJUDA1_,FRUTDA1_,BEANDAY_,GRENDAY_,ORNGDAY_,VEGEDA1_,_MISFRTN,_MISVEGN,_FRTRESP,_VEGRESP,_FRUTSUM,_VEGESUM,_FRTLT1,_VEGLT1,_FRT16,_VEG23,_FRUITEX,_VEGETEX,_TOTINDA,METVL11_,METVL21_,MAXVO2_,FC60_,ACTIN11_,ACTIN21_,PADUR1_,PADUR2_,PAFREQ1_,PAFREQ2_,_MINAC11,_MINAC21,STRFREQ_,PAMISS1_,PAMIN11_,PAMIN21_,PA1MIN_,PAVIG11_,PAVIG21_,PA1VIGM_,_PACAT1,_PAINDX1,_PA150R2,_PA300R2,_PA30021,_PASTRNG,_PAREC1,_PASTAE1,_LMTACT1,_LMTWRK1,_LMTSCL1,_RFSEAT2,_RFSEAT3,_FLSHOT6,_PNEUMO2,_AIDTST3"
    s2 = "44,2,2082015,2,8,2015,1100,2015000284,2015000284,1,1,,1,1,,1,0,1,,,,,,,,,2,88,88,,1,1,2,1,1,1,1,1,1,2,2,,2,2,2,2,2,2,3,,2,3,6,1,2,,2,2,7,88,99,2,118,501,,2,2,2,2,2,2,2,2,,,,3,888,,,,555,102,555,101,204,101,1,64,101,30,73,106,200,888,,,,,1,1,112014,5,1,2,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,2,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,5,5,,,1,35,,,,,,,2,1,,,,,,,,,,,10,1,2,441011,8.278774782802406,1.0,8.278774782802406,9,,,,9,,22.601497203099736,1,9,2,1,2,1,1,3,2,1,1,2,1,1,1,1,13,2,80,6,61,1.55,53.52,22.3,2,1,1,4,9,4,1,2,0,1,0,1,0.0,2.0,0.0,1.0,0.57,1.0,0,0,1,1,2.0,2.57,1,1,1,1,0,0,1,3.5,3.3,14.7,2.52,2,2,30,120,1.0,6.0,30,720,0.0,0,60,1440,1500,30,720,750,1,1,1,1,1,2,2,2,3,3,4,1,1,1,1,2"
    splits = s.split(sep=",")
    splits2 = s2.split(sep=",")
    for i,(s1, s2) in enumerate(zip(splits, splits2)):
        print(f"{i} : {s1} ({s2})")