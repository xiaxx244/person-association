import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

default_color = ['r','g','b','c','m','y','orange','brown']
default_marker = ['*','o','s','v','X','*','.','P']

class CMC:
    def __init__(self,cmc_dict, color=default_color, marker = default_marker):
        self.color = color
        self.marker = marker
        self.cmc_dict = cmc_dict

    def plot(self,title,rank=20, xlabel='Rank',ylabel='Matching Rates (%)',show_grid=True):
        fig, ax = plt.subplots()
        fig.suptitle(title)
        x = list(range(0, rank+1, 5))
        plt.ylim(0, 1.0)
        plt.xlim(1, rank)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x)
        plt.grid(show_grid)

        method_name = []
        i = 0
        for name in self.cmc_dict.keys():
            if rank < len(self.cmc_dict[name]):
                temp_cmc = self.cmc_dict[name][:rank]
                r = list(range(1, rank+1))
            else:
                temp_cmc = self.cmc_dict[name]
                r = list(range(1, len(temp_cmc)+1))

            if name == list(self.cmc_dict.keys())[-1]:
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[0], marker=self.marker[0], label='{:.1f}% {}'.format(self.cmc_dict[name][0]*100, name))
            else:
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[i+1], marker=self.marker[i+1], label='{:.1f}% {}'.format(self.cmc_dict[name][0]*100, name))
                i = i+1
            ax.add_line(globals()[name])
            method_name.append(globals()[name])

        plt.legend(handles=method_name)
        plt.show()

    def save(self, title, filename,
             rank=20, xlabel='Rank',
             ylabel='Matching Rates (%)', show_grid=True,
             save_path=os.getcwd(), format='png', **kwargs):
        fig, ax = plt.subplots()
        fig.suptitle(title)
        x = list(range(0, rank+1, 5))
        plt.ylim(0, 1.0)
        plt.xlim(1, rank)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x)
        plt.grid(show_grid)

        method_name = []
        i = 0
        for name in self.cmc_dict.keys():
            if rank < len(self.cmc_dict[name]):
                temp_cmc = self.cmc_dict[name][:rank]
                r = list(range(1, rank+1))
            else:
                temp_cmc = self.cmc_dict[name]
                r = list(range(1, len(temp_cmc)+1))

            if name == list(self.cmc_dict.keys())[-1]:
                globals()[name] = mlines.Line2D(r, temp_cmc, color='r', marker='*', label='{:.1f}% {}'.format(self.cmc_dict[name][0]*100, name))
            else:
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[i], marker=self.marker[i], label='{:.1f}% {}'.format(self.cmc_dict[name][0]*100, name))
                i = i+1
            ax.add_line(globals()[name])
            method_name.append(globals()[name])

        plt.legend(handles=method_name)
        fig.savefig(os.path.join(save_path,filename+'.'+format),
                    format=format,
                    bbox_inches='tight',
                   pad_inches = 0, **kwargs)

def mean_ap(true_score):
    count=0
    rank=[1,5,10,15,20,25]
    mAp=[]
    for i in range(len(true_score)):
        if true_score[i]==1:
            count=count+1
        if i in rank:
            mAp.append(count/i)
    return mAp
def main():
    cmc_dict ={
    'alignedReID': [0.10, 0.21, 0.29, 0.34, 0.40, 0.44, 0.47, 0.51, 0.54, 0.57],
    'deepReID': [0.65, 0.75, 0.81, 0.85, 0.89, 0.90, 0.91, 0.91, 0.93, 0.94],
    'triple_lossReD': [0.65, 0.75, 0.81, 0.85, 0.89, 0.90, 0.91, 0.91, 0.93, 0.94]
    }
    cmc = CMC(cmc_dict)
    #custimised color and marker
    new_color = ['r','g','b','c','m','y','orange','brown']
    new_marker = ['*','o','s','v','X','*','.','P']
    cmc = CMC(cmc_dict,color=new_color,marker=new_marker)
    cmc.save(title = 'CMC on CUHK01', filename='cmc_result_2',
         rank=10,xlabel='Rank Score', ylabel='Recognition Rate',
         show_grid=False, format='png')

main()
