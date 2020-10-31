#Create line graph of loss chart. x-axis = # of epochs, y-axis = loss
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

train_num = 940+2350 #760 for flickr, 1880 for mscoco outdoor decoder
val_num = 90+220 #140 for flickr, 180 for mscoco outdoor decoder
x_data_train = np.arange(1, 21, step=20/train_num) #1, 21 normally
x_data_val = np.arange(1, 21, step=20/val_num)
x_data_avgval = np.arange(1, 21, step=1)
print('x_data', x_data_train)
y_data_train = np.zeros((train_num)) 
y_data_val = np.zeros((val_num)) 
y_data_avgval = np.zeros((20))

#TOCHANGE:
infile = "output_mscoco_encoder_finetune.txt" #"output_outdoor.txt"
outfile = "loss_mscoco_outdoor_encoder_finetune.png" #"loss_mscoco_outdoor_decoder.png"
with open("output_outdoor.txt", "r") as output:
    train_i=0
    val_i=0
    valavg_i=0
    for line in output:
        if line.startswith("Epoch:"):
            splitLine = line.split("Loss ")
            try:
                splitLine = splitLine[1].split(" ")
            except:
                print(splitLine)
            loss = float(splitLine[0])
            #print(loss)
            y_data_train[train_i] = loss
            train_i+=1
        if line.startswith("Validation:"):
            splitLine = line.split("Loss ")
            try:
                splitLine = splitLine[1].split(" ")
            except:
                print(splitLine)
            loss = float(splitLine[0])
            #print(loss)
            y_data_val[val_i] = loss
            val_i+=1
        if line.startswith(" * LOSS"):
            splitLine = line.split("LOSS - ")
            try:
                splitLine = splitLine[1].split(",")
            except:
                print(splitLine)
            loss = float(splitLine[0])
            print('LOSS', loss)
            y_data_avgval[valavg_i] = loss
            valavg_i+=1
            
        if valavg_i == 10: break #FOR ENCODER FINETUNE GRAPH ONLY
            
    print(valavg_i)

with open(infile, "r") as output:
    for line in output:
        if line.startswith("Epoch:"):
            splitLine = line.split("Loss ")
            try:
                splitLine = splitLine[1].split(" ")
            except:
                print(splitLine)
            loss = float(splitLine[0])
            #print(loss)
            y_data_train[train_i] = loss
            train_i+=1
        if line.startswith("Validation:"):
            splitLine = line.split("Loss ")
            try:
                splitLine = splitLine[1].split(" ")
            except:
                print(splitLine)
            loss = float(splitLine[0])
            #print(loss)
            y_data_val[val_i] = loss
            val_i+=1
        if line.startswith(" * LOSS"):
            splitLine = line.split("LOSS - ")
            try:
                splitLine = splitLine[1].split(",")
            except:
                print(splitLine)
            loss = float(splitLine[0])
            print('LOSS', loss)
            y_data_avgval[valavg_i] = loss
            valavg_i+=1
                        
    print(valavg_i)
            

# utility function: color a string
def color(s, color=None, lightness=0):
              # red green brown blue purple cyan gray
              colors = None, "r", "g", "br", "b", "p", "c", "gy"
              esc = "\033[%d;3%dm" % (lightness, colors.index(color))
              return esc + s + "\033[0m"
        
# Define a function for the line plot
def lineplot(x_label, y_label, title):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data_train, y_data_train, lw = 1, color = '#539caf', alpha = 1)
    ax.plot(x_data_val, y_data_val, lw=1, color = 'r', alpha = 1)
    ax.plot(x_data_avgval, y_data_avgval, lw=1, color='g', alpha=1)
    #ax.plot(x_data, lasso_y, lw=2, color = 'r', alpha = 1)
    #ax.fill_between(x_data, low_CIs, upper_CIs, color = '#539caf', alpha = 0.4, label = '95% CI for LinUCB')
    #ax.fill_between(x_data, lasso_low, lasso_upper, color = 'r', alpha = 0.4, label = '95% CI for Lasso')
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.legend(('train loss', 'validation loss', 'average validation loss'))
    plt.show()
    plt.savefig('./lossgraphs/'+outfile)

np.set_printoptions(linewidth=999, edgeitems=10, suppress=True)
#warnings.filterwarnings("error")
np.random.seed(0)

lineplot(x_label = 'epochs', y_label = 'loss', title = 'MSCOCO Outdoor Encoder Finetuning Loss')

    
    