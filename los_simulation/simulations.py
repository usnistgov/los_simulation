# -----------------------------------------------------------------------------
# Name:   simulation.py     
# Purpose:    A simple line of site power simulator
# Authors:     aric.sanders@nist.gov
# Created:     03/19/205
# License:     NIST License
# -----------------------------------------------------------------------------
"""simulation.py is a module that defines transmitter and receiver nodes, and then given information about
location, direction and antenna pattern in 2 dimensions, calculates the linear sum of power at a location. This
simulation assumes line of sight propagation and uses the Friis equation to estimate power at a particular
point in space. This simulation is meant as a demonstration only.
"""
# -----------------------------------------------------------------------------
# Standard Imports
import sys
import os

# -----------------------------------------------------------------------------
# Third Party Imports
sys.path.append(os.path.join(os.path.dirname( __file__ ),'..')) # if the repo is library/source/modules, one deep change to .
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import EngFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import ndimage
from matplotlib.gridspec import GridSpec
# -----------------------------------------------------------------------------
# Module Constants
PARENT_DIRECTORY = os.path.dirname( __file__ )
TOWER_IMAGE_PATH = os.path.join(PARENT_DIRECTORY,"resources","tower.png")
WIFI_IMAGE_PATH = os.path.join(PARENT_DIRECTORY,"resources","wifi.png")
DISH_IMAGE_PATH = os.path.join(PARENT_DIRECTORY,"resources","dish.png")
C = 299792458 # speed of light

# -----------------------------------------------------------------------------
# Module Functions
def simple_directional_gain(theta_array,theta1=-(np.pi/180)*30,theta2=(np.pi/180)*30,gain1=9,gain2=-100): 
    """Creates a directive gain profile with gain1 from theta1 to theta2 and gain2 in all other directions"""
    return np.where((theta_array>=theta1)&(theta_array<=theta2),gain1,gain2)

def omni(theta_array,gain=9):
    """Returns an azimuthally symmetric profile with gain"""
    return gain*np.ones(theta_array.size)

def cos_three_halves(theta_array,gain=1):
    """Returns gain + cosine^3/2(theta) function """
    return gain+10*np.log10(np.cos(theta_array)**(3/2))

def ntia_very_high_gain_model_point(theta,gain =50):
    """From NTIA / ITS TM 09-461, A statistical gain antenna model that determines the radar antenna gain in the azimuth orientation. Assumes a maximum
    gain greater than and 48 dBi"""
    assert gain>48, "gain is too low to use this model, ntia_high_gain or ntia_medium_gain"
    theta = abs(theta)
    theta_m = 50*(.25*gain+7)**(.5)/10**(gain/20)
    theta_r = 27.466*10**(-.3*gain/10)
    theta_b = 48
    gain_out  = 0
    if 0<=theta<np.pi/180 *theta_m:
        gain_out = gain - 4*10**(-4)*(10**(gain/10))*(theta*180/np.pi)**2
    elif np.pi/180 *theta_m<=theta<np.pi/180 *theta_r:
        gain_out = .75*gain - 7
    elif np.pi/180 *theta_r<=theta<np.pi/180 *theta_b:
        gain_out = 29-25*np.log10(theta*180/np.pi)
    else:
        gain_out=-13
    return gain_out

ntia_very_high_gain_model = np.vectorize(ntia_very_high_gain_model_point,excluded=set(["gain"]),
                                         doc="Vectorized version of ntia_very_high_gain_model_point, function is for use as antenna pattern.")

 

def ntia_high_gain_model_point(theta,gain =30):
    """From NTIA / ITS TM 09-461, A statistical gain antenna model that determines the radar antenna gain in the azimuth orientation. Assumes a maximum
    gain between 22 dBi and 48 dBi"""
    assert 22<=gain<48, "gain is too low or too high to use this model, try ntia_very_high_gain or ntia_medium_gain"
    theta = abs(theta)
    theta_m = 50*(.25*gain+7)**(.5)/10**(gain/20)
    theta_r = 250/10**(gain/20)
    theta_b = 48
    gain_out  = 0
    if 0<=theta<np.pi/180 *theta_m:
        gain_out = gain - 4*10**(-4)*(10**(gain/10))*(theta*180/np.pi)**2
    elif np.pi/180 *theta_m<=theta<np.pi/180 *theta_r:
        gain_out = .75*gain - 7
    elif np.pi/180 *theta_r<=theta<np.pi/180 *theta_b:
        gain_out = 29-25*np.log10(theta*180/np.pi)
    else:
        gain_out=-13
    return gain_out

ntia_high_gain_model =np.vectorize(ntia_high_gain_model_point,excluded=set(["gain"]),
                                   doc="Vectorized version of ntia_high_gain_model_point, function is for use as antenna pattern.")

def ntia_medium_gain_model_point(theta,gain =20 ):
    """From NTIA / ITS TM 09-461, A statistical gain antenna model that determines the radar antenna gain in the azimuth orientation. Assumes a maximum
    gain between 10 dBi and 22 dBi"""
    assert 10<=gain<22, "gain is too low or too high to use this model, try ntia_very_high_gain or ntia_medium_gain"
    theta = abs(theta)
    theta_m = 50*(.25*gain+7)**(.5)/10**(gain/20)
    theta_r = 250/10**(gain/20)
    theta_b = 131.8257 * 10**(-gain/50)
    gain_out  = 0
    if 0<=theta<np.pi/180 *theta_m:
        gain_out = gain - 4*10**(-4)*(10**(gain/10))*(theta*180/np.pi)**2
    elif np.pi/180 *theta_m<=theta<np.pi/180 *theta_r:
        gain_out = .75*gain - 7
    elif np.pi/180 *theta_r<=theta<np.pi/180 *theta_b:
        gain_out = 53-gain/2-25*np.log10(theta*180/np.pi)
    else:
        gain_out = 0
    return gain_out

ntia_medium_gain_model=np.vectorize(ntia_medium_gain_model_point,excluded=set(["gain"]),
    doc="Vectorized version of ntia_medium_gain_model_point, function is for use as antenna pattern.")

def calculate_horizon(height):
    """Calculates the horizon distance in m given the height in meters"""
    return 3.56972*10**3*np.sqrt(height)


def calculate_relative_angle(x1,y1,x2,y2):
        """Calculates the relative angle between two locations specified as x1,y1 and x2,y2"""
        mag_a = np.sqrt(x2**2+y2**2)
        mag_b = np.sqrt(x1**2+y1**2)
        inverse_cos =  np.arccos((x1*x2+y1*y2)/(mag_a*mag_b))
        v1 = [x1,y1,0]
        v2= [x2,y2,0]
        cross_product =np.cross(v1,v2)
        if cross_product[2]<0:
            angle = -inverse_cos
        else:
            angle = inverse_cos
        return angle

vcal= np.vectorize(calculate_relative_angle,excluded=set(["x1","y1"]))
    #"Vectorized version of calculate_relative_angle"

        
def node_distance(node1,node2):
    "returns the distance between node1 and node2"
    return np.sqrt((node1.location[0]-node2.location[0])**2+(node1.location[1]-node2.location[1])**2)

def node_to_node_power(node1,node2,wavelength = C/3.75e9):
    """Returns the loss in dB between 2 nodes for a specified wavelength"""
    distance = node_distance(node1,node2)
    rx_angle = node1.calculate_relative_angle(node2.location[0]-node1.location[0],node2.location[1]-node1.location[1])
    tx_angle = node2.calculate_relative_angle(node1.location[0]-node2.location[1],node1.location[1]-node2.location[1])
    gain_rx = node1.antenna_pattern(rx_angle)
    gain_tx = node2.antenna_pattern(tx_angle)
    power_tx = node2.power
    power_rx = power_tx + gain_rx+ gain_tx + 20*np.log10(wavelength/(4*np.pi*distance))
    if isinstance(power_rx,np.ndarray):
        power_rx = power_rx[0]
    return power_rx

def node_to_node_loss(node1,node2,wavelength = C/3.75e9):
    """Returns the loss in dB between 2 nodes"""
    distance = node_distance(node1,node2)
    rx_angle = node1.calculate_relative_angle(node2.location[0]-node1.location[0],node2.location[1]-node1.location[1])
    tx_angle = node2.calculate_relative_angle(node1.location[0]-node2.location[1],node1.location[1]-node2.location[1])
    gain_rx = node1.antenna_pattern(rx_angle)
    gain_tx = node2.antenna_pattern(tx_angle)
    loss = gain_rx+ gain_tx + 20*np.log10(wavelength/(4*np.pi*distance))
    return loss

def fig2data(figure):
    "Returns a np array given a matplotlib figure"
    figure.canvas.draw()
    w, h = figure.canvas.get_width_height()
    buffer = np.frombuffer(figure.canvas.tostring_argb(), dtype=np.uint8)
    buffer.shape = (w, h, 4)
    buffer = np.roll(buffer, 3, axis=2)
    return buffer

def create_tower_glyph(percentage,rx_names=None,format_1={"color":"r"},format_2={"color":"b"},figsize = (5,5),fontsize =18,base_image = TOWER_IMAGE_PATH):
    """Creates a tower glyph with a percentage bar underneath, if percentage is a list then it auto adds bars"""
    if isinstance(percentage,(list,np.ndarray)):
        number_bars = len(percentage)
    else:
        number_bars = 1
                  
    fig = plt.figure( figsize=figsize,layout='constrained')
    gs  = GridSpec(5+number_bars,1) 
    ax1 = fig.add_subplot(gs[0:5,0:])

    ax1.imshow(plt.imread(base_image))
    ax1.set_axis_off()
    if isinstance(percentage,(list,np.ndarray)):
        for i in range(number_bars):
            if rx_names:
                name = rx_names[i]
            else:
                name = "A"
            if isinstance(format_1,list):
                format_1_  =format_1[i]
                format_2_ =format_2[i]
            else:
                format_1_  =format_1
                format_2_  =format_2
            bar_ax = fig.add_subplot(gs[5+i,0:])
            bar_ax.barh([name],[percentage[i]],**format_1_)
            bar_ax.barh([name],[1-percentage[i]],left = percentage[i],**format_2_)
            bar_ax.set_xticks([])
            bar_ax.set_yticks([])
            bar_ax.set_xlim([0,1])
            if rx_names:
                bar_ax.set_ylabel(name,fontsize=fontsize)
    else:
        if rx_names:
            name = rx_names
        else:
            name = "A"
        if isinstance(format_1,list):
            format_1_  =format_1[i]
            format_2_ =format_2[i]
        else:
            format_1_  =format_1
            format_2_  =format_2
        bar_ax = fig.add_subplot(gs[5,0:])    
        bar_ax.barh([name],[percentage],**format_1_)
        bar_ax.barh([name],[ 1-percentage],left = percentage,**format_2_)
        bar_ax.set_xticks([])
        bar_ax.set_yticks([])
        bar_ax.set_xlim([0,1])
        if rx_names:
            bar_ax.set_ylabel(name,fontsize=fontsize)
    fig.patch.set_alpha(0)

    plt.tight_layout()
    return fig2data(figure=fig)
# -----------------------------------------------------------------------------
# Module Classes
class Node():
    """A simple model of a receiver or transmitter that has location, direction, antenna pattern and an optional id.
      Has a convenience option for calculating the relative angle to any point, and also an azimuthal plot of the antenna function"""
    def __init__(self,location,direction=[0,1],antenna_pattern=simple_directional_gain,id=None):
        self.direction = direction
        self.location = location
        self.antenna_pattern = antenna_pattern
        self.id = id

    def calculate_relative_angle(self,x,y):
        return vcal(self.direction[0],self.direction[1],x,y)
    
    def plot_antenna_pattern(self,**options):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        theta = np.linspace(-np.pi, np.pi, 100)
        relative_theta = self.calculate_relative_angle(np.cos(theta),np.sin(theta))
        r = self.antenna_pattern(relative_theta)
        theta_direction = np.arctan2(self.direction[1],self.direction[0])
        ax.plot(theta_direction+relative_theta,r,color="b",marker="D",linestyle="solid")
        ax.set_title(f"Antenna Pattern Gain", va='bottom')
        ax.grid(True)



class TxNode(Node):
    "Transmitter node with power and an optional signal"
    def __init__(self,location,direction=[0,1],power = 1,antenna_pattern=simple_directional_gain,signal=1,id=None):
        super().__init__(direction=direction,location=location,antenna_pattern=antenna_pattern,id=id)
        self.signal = signal
        self.power = power
        
    def calculate_relative_angle(self,x,y):
        return vcal(self.direction[0],self.direction[1],x,y)

        
    
class RxNode(Node):
    def __init__(self,direction,location,antenna_pattern=simple_directional_gain,id=None):
        super().__init__(direction=direction,location=location,antenna_pattern=antenna_pattern,id=id)


# -----------------------------------------------------------------------------
# Module Scripts
def test_antenna_function(antenna_function =ntia_very_high_gain_model ):
    rx = RxNode(location=[0,0],direction=[0,1],antenna_pattern=antenna_function)
    rx.plot_antenna_pattern()
    plt.show()

def plot_antenna_functions(antenna_functions=[omni,simple_directional_gain,
                                              cos_three_halves,
                                              ntia_very_high_gain_model,ntia_high_gain_model,
                                              ntia_medium_gain_model],show =True,save=False):
    """plots all antenna functions in antenna_functions"""
    figure,ax = plt.subplots(subplot_kw={'projection': 'polar'})
    theta = np.linspace(-1*np.pi,1*np.pi,1000)
    for func in antenna_functions:
        try:
            ax.plot(theta,func(theta),label = func.__name__)
        except:
            raise
    ax.legend(loc='lower left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save:
        if isinstance(save,str):
            save_path = save
        else:
            save_path = "antena_functions.png"
        plt.savefig(save_path)
    if show:
     plt.show()

def show_tower_glyph(base_image=WIFI_IMAGE_PATH,show =True,save=False):
    """Shows the tower glyph, meant as a test of functionality"""
    im =create_tower_glyph([.5,.25],["Omni","Directional"],format_1=[{"color":"r","hatch":"/"},{"color":"g","hatch":"*"}],
                           format_2=[{"color":"b"},{"color":"y"}],fontsize=10,base_image=base_image)
    plt.imshow(im)
    plt.axis("off")
    if save:
        if isinstance(save,str):
            save_path = save
        else:
            save_path = "tower_glyph.png"
        plt.savefig(save_path)
    if show:
     plt.show()

def create_scenario_1(number_tx = 10,mean_tx_spacing=1000,relative_tx_power =76,wavelength = C/3.75e9 ,show =True,save=False):
    """Creates a simple scenario of 2 receivers at the origin, one is an omni directional and the other is a simple directional pointed in the 1i+1j direction. The number of transmitters
    are placed at random locations determined by mean_tx_spacing * uniform([-1,1]) in x and y amd are all omni directional emitters with a relative_tx_power. The 
    total power is calculated using a linear summation of powers for the wavelength using the friis formula. To save the image either specify save = path or save = True."""

    rx1 = RxNode(location=[0,0],direction=[0,1],antenna_pattern=omni,id="omni")
    rx2 = RxNode(location=[0,0],direction=[1,1],antenna_pattern=simple_directional_gain,id="directional")
    rxs  = [rx1,rx2]
    txs = []
    for i in range(number_tx):
        direction = [0,1]
        location = [mean_tx_spacing*np.random.uniform(low=-1, high=1),mean_tx_spacing*np.random.uniform(low=-1, high=1)]
        new_tx = TxNode(direction=direction,location=location,id=f"{i}",antenna_pattern=omni,power=relative_tx_power)
        txs.append(new_tx)
    fig,ax = plt.subplots()
    for rx in rxs:
        ax.plot(*rx.location,"rD")
        ax.quiver(rx.location[0],rx.location[1], rx.direction[0], rx.direction[1], color='r')
    for tx in txs:
            ax.plot(*tx.location,"gs")
            ax.quiver(tx.location[0],tx.location[1], tx.direction[0], tx.direction[1], color='g')
            ax.add_patch(patches.Circle(xy=tx.location,radius=100,color='g',alpha=1,fill=False))
    plt.grid()
    power_list_rx1 = np.array(list(map(lambda x: node_to_node_power(rx1,x,wavelength=wavelength),txs)))
    power_list_rx2 = np.array(list(map(lambda x: node_to_node_power(rx2,x,wavelength=wavelength),txs)))
    total_power_rx1 = 10*np.log10(np.sum(10**(power_list_rx1/10)))
    total_power_rx2 = 10*np.log10(np.sum(10**(power_list_rx2/10)))
    max_rx1 = txs[np.argmax(power_list_rx1)]
    plt.plot(max_rx1.location[0],max_rx1.location[1],"k.",markersize=22)
    max_rx2 = txs[np.argmax(power_list_rx2)]
    ax = plt.gca()
    ax.annotate(f"max of {rx1.id}",
                    xy=(max_rx1.location[0], max_rx1.location[1]), xycoords='data',
                    xytext=(1.5, 1.5), textcoords='offset points')
    ax.annotate(f"max of {rx2.id}",
                    xy=(max_rx2.location[0], max_rx2.location[1]), xycoords='data',
                    xytext=(-1.5, -1.5), textcoords='offset points',color="b")
    plt.title(f"{rx1.id} Power:{total_power_rx1:3.2f} dBm, {rx2.id} Power :{total_power_rx2:3.2f} dBm")
    if save:
        if isinstance(save,str):
            save_path = save
        else:
            save_path = "scenario_1.png"
        plt.savefig(save_path)
    if show:
     plt.show()

def create_scenario_2(number_tx=10,randomize_direction=True,r_tower_min=0,
                      r_tower_max=100000,angle_tower_min = -1*(np.pi/180)*180,
                      angle_tower_max = (np.pi/180)*180,transmitter_antenna_patten=simple_directional_gain,relative_tx_power =76,
                      wavelength = C/3.75e9,show =True,save=False):
    
    """Creates a simple scenario of 2 receivers at the origin, one is an omni directional and the other is a simple directional pointed in the 1i+1j direction. The number of transmitters
    are placed at random locations between a  radius of r_tower_min and r_tower_max and an angle of angle_tower_min and angle_tower_max. Each transmitter has either has a randomized
    direction or is pointed in the [0,1] direction and has transmitter_antenna_patten with relative_tx_power.
    The total power is calculated using a linear summation of powers for the wavelength using the friis formula. 
    To save the image either specify save = path or save = True."""

    formatter0 = EngFormatter(unit='m')
    rx1 = RxNode(location=[0,0],direction=[0,1],antenna_pattern=omni,id="omni")
    rx2 = RxNode(location=[0,0],direction=[1,1],antenna_pattern=simple_directional_gain,id="directional")
    rxs  = [rx1,rx2]
    txs = []
    for i in range(number_tx):
        if randomize_direction:
            direction = [np.random.uniform(low=-1, high=1),np.random.uniform(low=-1, high=1)]
        else:
            direction = [0,1]
        r_random = np.random.uniform(low= r_tower_min,high=r_tower_max)
        angle_random = np.random.uniform(low= angle_tower_min,high=angle_tower_max)
        location = [r_random*np.cos(angle_random),r_random*np.sin(angle_random)]
        new_tx = TxNode(direction=direction,location=location,id=f"{i}",antenna_pattern=transmitter_antenna_patten,power=relative_tx_power)
        txs.append(new_tx)
    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    ax.add_patch(patches.Circle([0,0],radius=r_tower_min,fill=False,color='r'))
    ax.add_patch(patches.Circle([0,0],radius=r_tower_max,fill=False,color='r'))
    ax.xaxis.set_major_formatter(formatter0)
    ax.yaxis.set_major_formatter(formatter0)
    ax.tick_params(axis='x', labelrotation=45)
    for rx in rxs:
        ax.plot(*rx.location,"rD")
        ax.quiver(rx.location[0],rx.location[1], rx.direction[0], rx.direction[1], color='r')
    for tx in txs:
            ax.plot(*tx.location,"gs")
            ax.quiver(tx.location[0],tx.location[1], tx.direction[0], tx.direction[1], color='g')
            ax.add_patch(patches.Circle(tx.location,radius=r_tower_max/20,fill=False))
    plt.grid()
    power_list_rx1 = np.array(list(map(lambda x: node_to_node_power(rx1,x,wavelength=wavelength),txs)))
    power_list_rx2 = np.array(list(map(lambda x: node_to_node_power(rx2,x,wavelength=wavelength),txs)))
    total_power_rx1 = 10*np.log10(np.sum(10**(power_list_rx1/10)))
    total_power_rx2 = 10*np.log10(np.sum(10**(power_list_rx2/10)))
    max_rx1 = txs[np.argmax(power_list_rx1)]
    max_rx2 = txs[np.argmax(power_list_rx2)]
    ax = plt.gca()
    ax.annotate(f"max of {rx1.id}",
                    xy=(max_rx1.location[0], max_rx1.location[1]), xycoords='data',
                    xytext=(-30, 10), textcoords='offset points',color="k",bbox=dict(facecolor='k', alpha=0.1))
    ax.annotate(f"max of {rx2.id}",
                    xy=(max_rx2.location[0], max_rx2.location[1]), xycoords='data',
                    xytext=(-30, -10), textcoords='offset points',color="k",bbox=dict(facecolor='k', alpha=0.1))
    plt.title(f"{rx1.id} Power:{total_power_rx1:3.2f} dBm, {rx2.id} Power :{total_power_rx2:3.2f} dBm")
    plt.xlim([1.1*-r_tower_max,1.1*r_tower_max])
    if save:
        if isinstance(save,str):
            save_path = save
        else:
            save_path = "scenario_2.png"
        plt.savefig(save_path)
    if show:
     plt.show()

def create_scenario_3(number_tx=10,randomize_direction=False,r_tower_min=0,
                      r_tower_max=100000,angle_tower_min = -1*(np.pi/180)*180,
                      angle_tower_max = (np.pi/180)*180,theta1 =np.pi/180 * -10,
                      theta2 = np.pi/180*10,transmitter_antenna_patten=omni,relative_tx_power =76,
                      wavelength = C/3.75e9,show =True,save=False): 
    """Creates a simple scenario of 2 receivers at the origin, one is an omni directional and the other is a simple directional pointed in the 1i+1j direction. The number of transmitters
    are placed at random locations between a  radius of r_tower_min and r_tower_max and an angle of angle_tower_min and angle_tower_max. Each transmitter has either has a randomized
    direction or is pointed in the [0,1] direction and has transmitter_antenna_patten with relative_tx_power.
    The total power is calculated using a linear summation of powers for the wavelength using the friis formula. 
    To save the image either specify save = path or save = True. The major difference in scenario 2 and scenario 3 is graphical"""

    formatter0 = EngFormatter(unit='m')
    directional_ = lambda x: simple_directional_gain(x,theta1=theta1,theta2=theta2,gain1=15,gain2=-20)

    rx1 = RxNode(location=[0,0],direction=[0,1],antenna_pattern=omni,id="omni")
    rx2 = RxNode(location=[0,0],direction=[1,1],antenna_pattern=directional_,id="directional")
    rxs  = [rx1,rx2]
    txs = []

    for i in range(number_tx):
        if randomize_direction:
            direction = [np.random.uniform(low=-1, high=1),np.random.uniform(low=-1, high=1)]
        else:
            direction = [0,1]
        r_random = np.random.uniform(low= r_tower_min,high=r_tower_max)
        angle_random = np.random.uniform(low= angle_tower_min,high=angle_tower_max)
        location = [r_random*np.cos(angle_random),r_random*np.sin(angle_random)]
        new_tx = TxNode(direction=direction,location=location,id=f"{i}",antenna_pattern=transmitter_antenna_patten,power=relative_tx_power)
        txs.append(new_tx)
    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    ax.add_patch(patches.Circle([0,0],radius=r_tower_min,fill=False,color='r',linestyle="dashed"))
    ax.add_patch(patches.Circle([0,0],radius=r_tower_max,fill=False,color='r',linestyle="dashed"))
    ax.xaxis.set_major_formatter(formatter0)
    ax.yaxis.set_major_formatter(formatter0)
    ax.tick_params(axis='x', labelrotation=45)
    for rx in rxs:
        ax.plot(*rx.location,"rD",alpha=0)
        if rx.antenna_pattern != omni:
            relative_angle = -180/np.pi*calculate_relative_angle(1,0,*rx.direction)
            image = plt.imread(DISH_IMAGE_PATH)
            rotated_image = ndimage.rotate(image, angle=relative_angle)         
            imagebox = OffsetImage(rotated_image, zoom=0.03)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox,rx.location, frameon=False)
            ax.add_artist(ab)
            arc = patches.Wedge(rx.location,r_tower_max,theta1=theta1*180/np.pi-relative_angle,theta2 = theta2*180/np.pi-relative_angle, lw=2,color="y",alpha=.2)
            ax.add_patch(arc)
    for tx in txs:
            imagebox = OffsetImage(plt.imread(TOWER_IMAGE_PATH), zoom=0.2)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox,tx.location, frameon=False)
            ax.add_artist(ab)
    plt.grid()
    power_list_rx1 = np.array(list(map(lambda x: node_to_node_power(rx1,x,wavelength=wavelength),txs)))
    power_list_rx2 = np.array(list(map(lambda x: node_to_node_power(rx2,x,wavelength=wavelength),txs)))
    total_power_rx1 = 10*np.log10(np.sum(10**(power_list_rx1/10)))
    total_power_rx2 = 10*np.log10(np.sum(10**(power_list_rx2/10)))
    max_rx1 = txs[np.argmax(power_list_rx1)]
    max_rx2 = txs[np.argmax(power_list_rx2)]
    ax = plt.gca()
    ax.annotate(f"max of {rx1.id}",
                    xy=(max_rx1.location[0], max_rx1.location[1]), xycoords='data',
                    xytext=(-30, 30), textcoords='offset points',color="k",bbox=dict(facecolor='b', alpha=0.1))
    ax.add_patch(patches.Circle(max_rx1.location,radius=r_tower_max/10,fill=False,color="b"))
    ax.annotate(f"max of {rx2.id}",
                    xy=(max_rx2.location[0], max_rx2.location[1]), xycoords='data',
                    xytext=(-30, -30), textcoords='offset points',color="k",bbox=dict(facecolor='r', alpha=0.1))
    ax.add_patch(patches.Circle(max_rx2.location,radius=r_tower_max/10,fill=False,color="r"))

    plt.xlim([-r_tower_max,r_tower_max])
    if save:
        if isinstance(save,str):
            save_path = save
        else:
            save_path = "scenario_3.png"
        plt.savefig(save_path)
    if show:
     plt.show()

def create_scenario_4(number_tx=4,randomize_direction=False,r_tower_min=100000,
                      r_tower_max=100000,angle_tower_min = -1*(np.pi/180)*180,
                      angle_tower_max = (np.pi/180)*180,theta1 =np.pi/180 * -10,
                      theta2 = np.pi/180*10,transmitter_antenna_patten=omni,relative_tx_power =76,
                      wavelength = C/3.75e9,show =True,save=False):
    
    """Creates a simple scenario of 2 receivers at the origin, one is an omni directional and the other is a directional with a spread of theta1 to theta2 rotated through 360 degrees . The number of transmitters
    are placed at random locations between a  radius of r_tower_min and r_tower_max and an angle of angle_tower_min and angle_tower_max. Each transmitter has either has a randomized
    direction or is pointed in the [0,1] direction and has transmitter_antenna_patten with relative_tx_power. The transmitter at 
    The total power is calculated using a linear summation of powers for the wavelength using the friis formula. 
    To save the image either specify save = path or save = True."""
        
    formatter0 = EngFormatter(unit='m')
    directional_ = lambda x: simple_directional_gain(x,theta1=theta1,theta2=theta2,gain1=15,gain2=-20)
    max_circle_radius = r_tower_max/6
    rx1 = RxNode([0,1],[0,0],antenna_pattern=omni,id="omni")
    rx2 = RxNode([1,1],[0,0],antenna_pattern=directional_,id="directional")
    rxs  = [rx1,rx2]
    txs = []
    for i in range(number_tx):
        if randomize_direction:
            direction = [np.random.uniform(low=-1, high=1),np.random.uniform(low=-1, high=1)]
        else:
            direction = [0,1]
        r_random = np.random.uniform(low= r_tower_min,high=r_tower_max)
        if i == 0:
            r_random = r_tower_max/2
        angle_ = np.linspace(angle_tower_min,angle_tower_max,number_tx+1)[i]
        location = [r_random*np.cos(angle_),r_random*np.sin(angle_)]
        new_tx = TxNode(direction=direction,location=location,id=f"{i}",antenna_pattern=transmitter_antenna_patten,power=relative_tx_power)
        txs.append(new_tx)
        
    for i,rx_theta in enumerate(np.linspace(-np.pi,np.pi,36)):
            rx2.direction = [np.cos(rx_theta),np.sin(rx_theta)]
            fig = plt.figure(figsize=(10,10),constrained_layout=True)
            gs = GridSpec(4,5,figure=fig)
            ax  = fig.add_subplot(gs[0:,0:4])
            ax.set_aspect('equal')
            ax.xaxis.set_major_formatter(formatter0)
            ax.yaxis.set_major_formatter(formatter0)
            ax.tick_params(axis='x', labelrotation=45)
            for rx in rxs:
                    ax.plot(*rx.location,"rD",alpha=0)
                    if rx.antenna_pattern != omni:
                            relative_angle = -180/np.pi*calculate_relative_angle(1,0,*rx.direction)
                            image = plt.imread(DISH_IMAGE_PATH)
                            rotated_image = ndimage.rotate(image, angle=-90-relative_angle)         
                            imagebox = OffsetImage(rotated_image, zoom=0.03)
                            imagebox.image.axes = ax
                            ab = AnnotationBbox(imagebox,rx.location, frameon=False)
                            ax.add_artist(ab)
                            arc = patches.Wedge(rx.location,r_tower_max,theta1=theta1*180/np.pi-relative_angle,theta2 = theta2*180/np.pi-relative_angle, lw=2,color="y",alpha=.2)
                            ax.add_patch(arc)
            ax.grid()
            power_list_rx1 = np.array(list(map(lambda x: node_to_node_power(rx1,x,wavelength=wavelength),txs)))
            percentage_rx1 = 10**(power_list_rx1/10)/np.sum(10**(power_list_rx1/10))
            power_list_rx2 = np.array(list(map(lambda x: node_to_node_power(rx2,x,wavelength=wavelength),txs)))
            percentage_rx2 = 10**(power_list_rx2/10)/np.sum(10**(power_list_rx2/10))
            total_power_rx1 = 10*np.log10(np.sum(10**(power_list_rx1/10)))
            total_power_rx2 = 10*np.log10(np.sum(10**(power_list_rx2/10)))
            max_rx1 = txs[np.argmax(power_list_rx1)]
            max_rx2 = txs[np.argmax(power_list_rx2)]
            ax.add_patch(patches.Circle(max_rx1.location,radius=max_circle_radius,fill=True,color="b",alpha=.2))
            ax.add_patch(patches.Circle(max_rx1.location,radius=max_circle_radius,fill=False,color="b",alpha=1))
            ax.add_patch(patches.Circle(max_rx2.location,radius=.95*max_circle_radius,fill=True,color="r",alpha=.2))
            ax.add_patch(patches.Circle(max_rx2.location,radius=.95*max_circle_radius,fill=False,color="r",alpha=1))

            ax.annotate(f"Omni- Power:{total_power_rx1:3.2f} Total-Max: {total_power_rx1-max(power_list_rx1):3.2f}\nDirectional- Power: {total_power_rx2:3.2f} Total-Max: {total_power_rx2-max(power_list_rx2):3.2f}",
                            xy=(-r_tower_max,-r_tower_max-r_tower_max/5), xycoords='data',
                            xytext=(5, -10), textcoords='offset points',color="k",bbox=dict(facecolor='w', alpha=0.5))
            for tx_index,tx in enumerate(txs):
                    im = create_tower_glyph([percentage_rx1[tx_index],percentage_rx2[tx_index]],format_1=[{"color":"r","hatch":"/"},
                                                                                                          {"color":"g","hatch":"\\"}],format_2=[{"color":"b"},{"color":"w"}],fontsize=10,base_image=WIFI_IMAGE_PATH)
                    plt.close()
                    imagebox = OffsetImage(im, zoom=0.05)
                    imagebox.image.axes = ax
                    ab = AnnotationBbox(imagebox,tx.location, frameon=False)
                    ax.add_artist(ab)
            ax.set_xlim([1.1*-r_tower_max,1.1*r_tower_max])
            ax.set_ylim([1.1*(-r_tower_max -r_tower_max/4),1.1*r_tower_max])
            ax1  = fig.add_subplot(gs[0,4])
            ax1.set_axis_off()
            im_= create_tower_glyph([percentage_rx1[np.argmax(power_list_rx1)],percentage_rx2[np.argmax(power_list_rx1)]],["RX1","RX2"],
                                    format_1=[{"color":"r","hatch":"/"},{"color":"g","hatch":"\\"}],format_2=[{"color":"b"},{"color":"w"}],fontsize=22,base_image=WIFI_IMAGE_PATH);
            plt.close()
            ax1.imshow(im_)
            ax1.set_title(f"Max of Omni")
            ax2  = fig.add_subplot(gs[1,4])
            ax2.set_axis_off()
            im_= create_tower_glyph([percentage_rx1[np.argmax(power_list_rx2)],percentage_rx2[np.argmax(power_list_rx2)]],["RX1","RX2"],
                                    format_1=[{"color":"r","hatch":"/"},{"color":"g","hatch":"\\"}],format_2=[{"color":"b"},{"color":"w"}],fontsize=22,base_image=WIFI_IMAGE_PATH);
            plt.close()
            ax2.imshow(im_)
            ax2.set_title(f"Max of Directional")
            ax3 = fig.add_subplot(gs[2,4],**{'projection': 'polar'})
            theta = np.linspace(-np.pi, np.pi, 100)
            relative_theta = rx1.calculate_relative_angle(np.cos(theta),np.sin(theta))
            r = rx1.antenna_pattern(relative_theta)
            theta_direction = np.arctan2(rx1.direction[1],rx1.direction[0])
            ax3.plot(theta_direction+relative_theta,r,color="r",marker="D",linestyle="solid")
            ax3.set_title(f"Rx1 Antenna Pattern Gain", va='bottom')
            ax3.grid(True)
            ax4 = fig.add_subplot(gs[3,4],**{'projection': 'polar'})
            theta = np.linspace(-np.pi, np.pi, 100)
            relative_theta = rx2.calculate_relative_angle(np.cos(theta),np.sin(theta))
            r = rx2.antenna_pattern(relative_theta)
            theta_direction = np.arctan2(rx2.direction[1],rx2.direction[0])
            ax4.plot(theta_direction+relative_theta,r,color="g",marker="D",linestyle="solid")
            ax4.set_title(f"Rx2 Antenna Pattern Gain", va='bottom')
            ax4.grid(True)
            plt.tight_layout
            if save:
                if isinstance(save,str):
                    directory = os.path.dirname(save)
                    basename = os.path.basename(save)
                    save_path = os.path.join(directory,f"{i}_"+basename)
                else:
                    save_path = f"{i}_scenario_4.png"
                plt.savefig(save_path)
            if show:
                plt.show()            
            plt.close()


# -----------------------------------------------------------------------------
# Module Runner
if __name__=="__main__":
    test_antenna_function()
    #plot_antenna_functions()
    #show_tower_glyph()
    #create_scenario_1()
    #create_scenario_2()
    #create_scenario_3()
    #create_scenario_4()