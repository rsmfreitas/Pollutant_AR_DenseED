import h5py
import torch as th
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')

def plot_pred(samples_target, samples_output, epoch, idx, output_dir):

    samples_err = samples_target - samples_output
    samples = np.vstack((samples_target, samples_output, samples_err))

    Nout = samples_target.shape[0]
    c_max = np.full( (Nout*3), 0.0)
    c_min = np.full( (Nout*3), 0.0)
    for l in range(Nout*3):
        if l < Nout:
            c_max[l] = np.max(samples[l])
        elif Nout <= l < 2*Nout:
            c_max[l] = np.max(samples[l])
            # if c_max[l] > c_max[l-Nout]:
            #     c_max[l-Nout] = c_max[l]
            # else:
            #     c_max[l] = c_max[l-Nout]
        elif l >=  2*Nout:
            c_max[l] = np.max( np.abs(samples[l]) )
            c_min[l] = 0. - np.max( np.abs(samples[l]) )

    LetterId = (['a','b','c','d', 'e','f','g','h', 'i','j','k','m'])
    ylabel = (['$\mathbf{y}$', '$\hat{\mathbf{y}}$', '$\mathbf{y}-\hat{\mathbf{y}}$'])
    fig = plt.figure(figsize=(16, 10))
    outer = gridspec.GridSpec(2, 1, wspace=0.01, hspace=0.06)
    nl = 40
    m = 0
    samp_id = [ [0,1,2,3, 10,11,12,13, 20,21,22,23], [4,5,6,7, 14,15,16,17, 24,25,26,27] ]
    for j in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec = outer[j], wspace=0.2, hspace=0.08)
        l = 0
        for k in range(3*4):
            ax = plt.Subplot(fig, inner[k])
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            s_id = samp_id[j][k]
            cax = ax.imshow(samples[s_id], cmap='jet', origin='lower',vmin=c_min[s_id], vmax=c_max[s_id])
            fig.add_subplot(ax)
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            cbar = plt.colorbar(cax, ax=ax, fraction=0.021, pad=0.04,
                                format=ticker.FuncFormatter(lambda x, pos: "%.3f" % x ))
            cbar.ax.tick_params(labelsize=10)

            if k < 4:
                ax.text(65, 7, '$({})\ t={}$ [T]'.format(LetterId[m],(m+1)), fontsize=12,color='white')
                m = m + 1
            if np.mod(k,4) == 0:
                if j == 0:
                    ax.set_ylabel(ylabel[l], fontsize=14)
                    l = 1 + l
                else:
                    ax.set_ylabel(ylabel[l], fontsize=14)
                    l = 1 + l

    plt.savefig(output_dir + '/epoch_{}_output_{}.png'.format(epoch, idx),
                bbox_inches='tight',dpi=400)
    plt.close(fig)
    print("epoch {}, done with printing sample output {}".format(epoch, idx))


def plot_pred_ar(sample, samples_target, samples_output, ntimes, epoch, idx, output_dir):
    samples = np.vstack((samples_target, samples_output, samples_target - samples_output))

    column = ntimes #+ 1
    c_max = np.full( (column*3), 0.0) # the same color scale for the predicted output fields at the same time step
    for l in range(column*3):
        if l < column:
            c_max[l] = np.max(samples[l])
        elif column <= l < 2*column:
            c_max[l] = np.max(samples[l])
            if c_max[l] > c_max[l-column]:
                c_max[l-column] = c_max[l]
            else:
                c_max[l] = c_max[l-column]
        else:
            c_max[l] = np.max( np.abs(samples[l]) )

    LetterId = (['a','b','c','d', 'e','f','g','h', 'i','j','k','m'])
    ylabel = (['$\mathbf{y}$', '$\hat{\mathbf{y}}$', '$\mathbf{y}-\hat{\mathbf{y}}$'])
    fig = plt.figure(figsize=(16, 10))
    outer = gridspec.GridSpec(2, 1, wspace=0.01, hspace=0.06)
    nl = 40
    m = 0
    samp_id = [ [0,1,2,3, 10,11,12,13, 20,21,22,23], [4,5,6,7, 14,15,16,17, 24,25,26,27] ]
    for j in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec = outer[j], wspace=0.2, hspace=0.08)
        l = 0
        for k in range(3*4):
            ax = plt.Subplot(fig, inner[k])
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            s_id = samp_id[j][k]
            if k < 2*4:
                cax = ax.contourf(samples[ s_id,0], np.arange(0.0 , c_max[s_id] + c_max[s_id]/nl*1, c_max[s_id]/nl), cmap='jet')
                fig.add_subplot(ax)
            else:
                cax = ax.contourf(samples[ s_id,0], np.arange(0.0 - c_max[s_id] - c_max[s_id]/nl*1, c_max[s_id] + c_max[s_id]/nl*1, c_max[s_id]/nl), cmap='jet',extend='both')
                fig.add_subplot(ax)
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            cbar = plt.colorbar(cax, ax=ax, fraction=0.021, pad=0.04,
                                format=ticker.FuncFormatter(lambda x, pos: "%.3f" % x ))
            cbar.ax.tick_params(labelsize=10)

            if k < 4:
                ax.text(65, 7, '$({})\ t={}$ [T]'.format(LetterId[m],(m+1)), fontsize=14,color='white')
                m = m + 1
            if np.mod(k,4) == 0:
                if j == 0:
                    ax.set_ylabel(ylabel[l], fontsize=14)
                    l = 1 + l
                else:
                    ax.set_ylabel(ylabel[l], fontsize=14)
                    l = 1 + l

    plt.savefig(output_dir + '/epoch_{}_output_{}.png'.format(epoch, idx[sample]),
                bbox_inches='tight',dpi=400)
    plt.close(fig)
    print("epoch {}, done with printing sample output {}".format(epoch, idx[sample]))



def plot_r2_rmse(r2_train, r2_test, rmse_train, rmse_test, exp_dir, args):
    x = np.arange(args.log_interval, args.n_epochs + args.log_interval,
                args.log_interval)
    plt.figure()
    plt.plot(x, r2_train, 'k', label="train: {:.3f}".format(np.mean(r2_train[-5: -1])))
    plt.plot(x, r2_test, 'r', linestyle = '--', label="test: {:.3f}".format(np.mean(r2_test[-5: -1])))
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('$R^2$', fontsize=14)
    plt.legend(loc='lower right')
    plt.savefig(exp_dir + "/r2.png", dpi=400)
    plt.close()
    np.savetxt(exp_dir + "/r2_train.txt", r2_train)
    np.savetxt(exp_dir + "/r2_test.txt", r2_test)

    plt.figure()
    plt.plot(x, rmse_train, 'k', label="train: {:.3f}".format(np.mean(rmse_train[-5: -1])))
    plt.plot(x, rmse_test, 'r', linestyle = '--', label="test: {:.3f}".format(np.mean(rmse_test[-5: -1])))
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.legend(loc='upper right')
    plt.savefig(exp_dir + "/rmse.png", dpi=400)
    plt.close()
    np.savetxt(exp_dir + "/rmse_train.txt", rmse_train)
    np.savetxt(exp_dir + "/rmse_test.txt", rmse_test)
