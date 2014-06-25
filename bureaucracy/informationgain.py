
import numpy as np
#from numpy import log2, linspace, array, shape

def shannon(x):
    if isinstance(x, np.ndarray) == False: 
        if x>0:
            return x*np.log2(x)
        else:
            return 0.0        
        
    
    x[x == 0] = 1.0
    return x*np.log2(x)


def expectedInformationGain(p0, M_ll, M_nn):
    p1 = 1-p0
        
    I = p0 * (shannon(M_ll) + shannon(1-M_ll)) + \
        p1 * (shannon(M_nn) + shannon(1-M_nn)) - \
        shannon(M_ll*p0 + (1-M_nn)*p1) - \
        shannon((1-M_ll)*p0 + M_nn*p1)
    
    return I

def informationGain(p0, M_ll, M_nn, c):
    p1 = 1-p0
    
    if c:
        M_cl = M_ll
        M_cn = 1-M_nn
    else:
        M_cl = 1-M_ll
        M_cn = M_nn
        
    #oldI = -shannon(p0) -shannon(p1) - np.log2(M_cl*p0 + M_cn*p1) \
    #        + (1/(M_cl*p0 + M_cn*p1))*(p0*shannon(M_cl) + M_cl*shannon(p0) \
    #                                 + p1*shannon(M_cn) + M_cn*shannon(p1) )
    #print str(oldI)

    pc = M_cl*p0 + M_cn*p1
    
    #if isinstance(pc, np.ndarray):
    #    p0_c = pc
    #    p1_c = pc
        
    #    p0_c[pc!=0] = M_cl/pc[pc!=0]
    #    p1_c[pc!=0] = M_cn/pc[pc!=0]
    #elif pc==0:
    #    p0_c = 0
    #    p1_c = 0
    #else:
        
    p0_c = M_cl/pc
    p1_c = M_cn/pc
    
    #print str(p0)
    #print str(p1)
    #print "p0_c=" + str(p0_c*p0) + ", p1_c=" + str(p1_c*p1) + ", total=" + str(p0_c*p0+p1_c*p1)
    
    I = p0*shannon(p0_c) + p1*shannon(p1_c)
    print "Min info gain value (a rounding error?): " + str(np.min(I))

    return I

# =============================================================================================

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    print 'Calculating expected information gain for a volunteer with 0.75 correctness in both classes...'

    #Priors over the classes in a binary classification problem
    logp0 = np.linspace(-4, 0, 500)
    p0 = 10**logp0

    # 2x2 confusion matrices of some volunteers' agents:
    M_ll = [0.55,0.80,0.999,0.25,0.9999,0.500,0.999]
    M_nn = [0.55,0.80,0.999,0.25,0.0001,0.999,0.500]

    almostrandom = expectedInformationGain(p0, M_ll[0], M_nn[0]) 
    mediumastute = expectedInformationGain(p0, M_ll[1], M_nn[1])
    highlyastute = expectedInformationGain(p0, M_ll[2], M_nn[2]) 
    fairlyobtuse = expectedInformationGain(p0, M_ll[3], M_nn[3])
    optimistroll = expectedInformationGain(p0, M_ll[4], M_nn[4])
    slightpessimist = expectedInformationGain(p0, M_ll[5], M_nn[5])
    slightoptimist = expectedInformationGain(p0, M_ll[6], M_nn[6])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(p0, almostrandom, label=("Almost Random: ["+str(M_ll[0])+", "+str(M_nn[0])+"]") )
    ax.plot(p0, mediumastute, label=("Medium Astute: ["+str(M_ll[1])+", "+str(M_nn[1])+"]") )
    ax.plot(p0, highlyastute, label=("Highly Astute: ["+str(M_ll[2])+", "+str(M_nn[2])+"]") )
    ax.plot(p0, fairlyobtuse, label=("Fairly Obtuse: ["+str(M_ll[3])+", "+str(M_nn[3])+"]") )
    ax.plot(p0, optimistroll, label=("Optimist Troll: ["+str(M_ll[4])+", "+str(M_nn[4])+"]") )
    ax.plot(p0, slightpessimist, label=("Slight Pessimist: ["+str(M_ll[5])+", "+str(M_nn[5])+"]") )
    ax.plot(p0, slightoptimist, label=("Slight Optimist: ["+str(M_ll[6])+", "+str(M_nn[6])+"]") )
    ax.axvline(x=0.5, ls=':')
    ax.set_xscale('log')
    ax.set_title("Expected Information Gain")
    ax.set_xlabel("Pr(LENS)")
    ax.set_ylabel("Expected Information Gain from Agent (bits)")
    ax.legend(loc="best")
    #plt.show()
    fig.savefig('expectedinfogain.png')

    print 'Testing the contribution of volunteers after observing a classification'


    almostrandomTrue = informationGain(p0, M_ll[0], M_nn[0], True ) 
    mediumastuteTrue  = informationGain(p0, M_ll[1], M_nn[1], True )
    highlyastuteTrue  = informationGain(p0, M_ll[2], M_nn[2], True ) 
    fairlyobtuseTrue  = informationGain(p0, M_ll[3], M_nn[3], True )
    optimistrollTrue  = informationGain(p0, M_ll[4], M_nn[4], True )
    slightpessimistTrue  = informationGain(p0, M_ll[5], M_nn[5], True )
    slightoptimistTrue  = informationGain(p0, M_ll[6], M_nn[6], True )
    
    almostrandomFalse = informationGain(p0, M_ll[0], M_nn[0], False ) 
    mediumastuteFalse  = informationGain(p0, M_ll[1], M_nn[1], False )
    highlyastuteFalse  = informationGain(p0, M_ll[2], M_nn[2], False ) 
    fairlyobtuseFalse  = informationGain(p0, M_ll[3], M_nn[3], False )
    optimistrollFalse  = informationGain(p0, M_ll[4], M_nn[4], False )
    slightpessimistFalse  = informationGain(p0, M_ll[5], M_nn[5], False )
    slightoptimistFalse  = informationGain(p0, M_ll[6], M_nn[6], False )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(p0, almostrandomTrue, label=("Almost Random: ["+str(M_ll[0])+", "+str(M_nn[0])+"]") )
    ax.plot(p0, mediumastuteTrue, label=("Medium Astute: ["+str(M_ll[1])+", "+str(M_nn[1])+"]") )
    ax.plot(p0, highlyastuteTrue, label=("Highly Astute: ["+str(M_ll[2])+", "+str(M_nn[2])+"]") )
    ax.plot(p0, fairlyobtuseTrue, label=("Fairly Obtuse: ["+str(M_ll[3])+", "+str(M_nn[3])+"]") )
    ax.plot(p0, optimistrollTrue, label=("Optimist Troll: ["+str(M_ll[4])+", "+str(M_nn[4])+"]") )
    ax.plot(p0, slightpessimistTrue, label=("Slight Pessimist: ["+str(M_ll[5])+", "+str(M_nn[5])+"]") )
    ax.plot(p0, slightoptimistTrue, label=("Slight Optimist: ["+str(M_ll[6])+", "+str(M_nn[6])+"]") )
    ax.axvline(x=0.5, ls=':')    
 
    ax.set_xscale('log')
    ax.set_title("Agents Answer True.")
    ax.set_xlabel("p(LENS)")
    ax.set_ylabel("Contributed Information by an Agent (bits)")
    ax.legend(loc="best")
    fig.savefig('infogain_true.png')       
 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(p0, almostrandomFalse, label=("Almost Random: ["+str(M_ll[0])+", "+str(M_nn[0])+"]") )
    ax.plot(p0, mediumastuteFalse, label=("Medium Astute: ["+str(M_ll[1])+", "+str(M_nn[1])+"]") )
    ax.plot(p0, highlyastuteFalse, label=("Highly Astute: ["+str(M_ll[2])+", "+str(M_nn[2])+"]") )
    ax.plot(p0, fairlyobtuseFalse, label=("Fairly Obtuse: ["+str(M_ll[3])+", "+str(M_nn[3])+"]") )
    ax.plot(p0, optimistrollFalse, label=("Optimist Troll: ["+str(M_ll[4])+", "+str(M_nn[4])+"]") )
    ax.plot(p0, slightpessimistFalse, label=("Slight Pessimist: ["+str(M_ll[5])+", "+str(M_nn[5])+"]") )
    ax.plot(p0, slightoptimistFalse, label=("Slight Optimist: ["+str(M_ll[6])+", "+str(M_nn[6])+"]") )
    ax.axvline(x=0.5, ls=':')     
    
    #ax.plot(p0, sampleRes, label=("true, M_ll="+str(M_ll[0])+", M_nn="+str(M_nn[0])) )
    #ax.plot(p0, sampleResFalse, label=("false, M_ll="+str(M_ll[0])+", M_nn="+str(M_nn[0])) )
    ax.set_xscale('log')
    ax.set_title("Agents Answer False.")
    ax.set_xlabel("p(LENS)")
    ax.set_ylabel("Contributed Information by an Agent (bits)")
    ax.legend(loc="best")
    #plt.show()
    fig.savefig('infogain_false.png')
