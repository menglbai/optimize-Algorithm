import numpy as np
import matplotlib.pyplot as plt
# import dill
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
def Utility_and_ω_N():
    colors = ['mediumpurple', 'hotpink', 'mediumseagreen', 'cornflowerblue', 'orange' ]
    line_types = ['v-', 'd-','^-','.-','*-']
    plt.rc('font', family='serif')
    plt.rc('font', size=28)
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    plt.figure(figsize=(16, 6))
    grid = plt.GridSpec(1, 2, wspace=0.3, hspace=0.3)

    axes1 = plt.subplot(grid[0, 0])
    bwith = 2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    x = np.linspace(0.001,0.9,20)

    axes1.set_title('(a)', y=-0.3, fontsize='small')
    axes1.set_ylabel('Average Utility')
    axes1.set_xlabel('ω')

    L_MOODO = np.array([0.262534711,0.263799693,0.265008736,0.265792956,0.266281275,
                       0.266549027,0.266643931,0.266598052,0.266433871,0.266167678,
                       0.265811558,0.265374626,0.264863836,0.264284524,0.263640777,
                       0.262935687,0.262171549,0.261350001,0.260472109,0.259538605])
    L = np.zeros(20)
    F = np.array([-1.398703573,-1.508269501,-1.59620703,-1.670336096,-1.734801186,
         -1.792085244,-1.84380249,-1.894387852,-1.938258261,-1.979045829,
         -2.017216632,-2.053136525,-2.087098407,-2.119340751,-2.150060562,
         -2.179422672,-2.207566559,-2.234611448,-2.260660184,-2.285802229])
    G = np.array([0.231069918,0.229420714,0.228333446,0.226646134,0.225043587,
         0.223584827,0.22224305,0.211557689,0.209922353,0.208338621,
         0.206797266,0.205291535,0.203816924,0.202371146,0.200954961,
         0.19957555,0.198224278,0.196541037,0.194855088,0.193163566])
    R = np.array([-0.190679003,-0.266418646,-0.319362337,-0.358669098,-0.405034591,
         -0.444793508,-0.471663264,-0.491071045,-0.520378183,-0.543414683,
         -0.568475971,-0.59985494,-0.620521097,-0.633590892,-0.655885367,
         -0.676267283,-0.706866005,-0.705533716,-0.736497284,-0.744900983])

    line1, = axes1.plot(x, F, line_types[0], lw=2, label="F. Offl.",color=colors[0])
    line2, = axes1.plot(x, R, line_types[1], lw=2, label="R. Offl.",color=colors[1])
    line3, = axes1.plot(x, L, line_types[2], lw=2, label="L. Offl.",color=colors[2])
    line4, = axes1.plot(x, G, line_types[3], lw=2, label="G. Offl.", color=colors[3])
    line5, = axes1.plot(x, L_MOODO, line_types[4], lw=2, label="RM-OCO",color=colors[4])

    plt.legend(loc=3, prop={'size': 16},ncol=1)

    # 子图2
    axes2 = plt.subplot(grid[0, 1])
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    axes2.set_title('(b)', y=-0.3, fontsize='small')
    axes2.set_ylabel('Average Utility')
    axes2.set_xlabel('Number of users')

    Ns = [1, 2, 5, 15, 25, 35, 45, 55, 65, 75, 85, 100]
    L_MOODO = np.array([2.328816033, 1.947532076, 1.206692781, 0.459031983, 0.267207231,
                        0.182130822, 0.138205205, 0.109151788, 0.089981571, 0.076747769,
                        0.066147508, 0.054706305])
    L = np.zeros(12)
    F = np.array([2.318979885, 1.932249307, 0.871703159, -1.065093047, -2.036730373,
                  -2.380031107, -2.483302586, -2.523678504, -2.522233099, -2.548788366,
                  -2.536447295, -2.538993241])
    G = np.array([2.31957765, 1.933489143, 1.084664855, 0.361729919, 0.216122779,
                  0.15057793, 0.11580008, 0.091781629, 0.076554625, 0.065274924,
                  0.05635218, 0.047450699])
    R = np.array([2.010670593, 1.739401167, 1.05934424, 0.207648966, -0.615342474,
                  -1.175160807, -1.540544988, -1.810748758, -1.923460345, -2.012498333,
                  -2.052265776, -2.093198862])

    line1, = axes2.plot(Ns, F, line_types[0], lw=2, label="F. Offl.", color=colors[0])
    line2, = axes2.plot(Ns, R, line_types[1], lw=2, label="R. Offl.", color=colors[1])
    line3, = axes2.plot(Ns, L, line_types[2], lw=2, label="L. Offl.", color=colors[2])
    line4, = axes2.plot(Ns, G, line_types[3], lw=2, label="G. Offl.", color=colors[3])
    line5, = axes2.plot(Ns, L_MOODO, line_types[4], lw=2, label="RM-OCO", color=colors[4])

    path_name = "comparion_method1"
    plt.legend(loc=1, prop={'size': 16},ncol=1)
    plt.savefig("plots/" + path_name + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)

Utility_and_ω_N()


