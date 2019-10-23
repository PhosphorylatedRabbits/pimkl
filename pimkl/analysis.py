import sys
import io
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import betainc
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection, multipletests

SIGNIFICANT_COLOR = sns.color_palette('colorblind')[2]
OTHER_COLOR = sns.color_palette('colorblind')[0]


def significant_pathways(df, alpha=1e-3):
    """significance test vs the average weight"""
    avg = 1. / df.shape[1]
    ordering = df.columns
    wilcoxon_pvalues = pd.Series(
        {
            ws: wilcoxon(df[ws] - avg)[1] if df[ws].median() > avg else 1.
            for ws in df
        }
    )
    wilcoxon_pvalues = wilcoxon_pvalues[ordering]
    # Multiple Testing Correction
    reject_H0, pvals_corrected = fdrcorrection(wilcoxon_pvalues, alpha=alpha)
    return pd.Series(reject_H0, index=ordering)


def significant_correlations(rho, sample_length):
    triu_indices = np.triu_indices(rho.shape[0], 1)
    rhof = rho.values[triu_indices]
    dof = sample_length - 2
    ts = rhof * rhof * (dof / (1 - rhof * rhof + sys.float_info.epsilon))
    p_values = betainc(0.5 * dof, 0.5, dof / (dof + ts))
    significants, p_values_corrected, _, _ = multipletests(
        p_values, alpha=5e-2, method='fdr_bh'
    )
    p_vals = np.zeros_like(rho)
    p_vals[triu_indices] = p_values_corrected
    p_vals += p_vals.T
    sig_mask = np.zeros_like(rho, dtype=np.bool)
    sig_mask[triu_indices] = significants
    sig_mask += (sig_mask.T + np.eye(rho.shape[0], dtype=np.bool))
    return sig_mask, p_vals


def plot_aucs_to_buffer(df, save=False):
    """
    plot AUC for multiindexed pandas.DataFrame where
    df.columns.names = ['data', 'kind']
    """
    buffer = None
    aucs_df_box = df.melt(value_name='AUC')
    if 'class' in df.columns.levels[1]:
        sns.boxplot(data=aucs_df_box, y='AUC', x='data')
    else:
        sns.boxplot(
            data=aucs_df_box, y='AUC', x='kind', hue='data'
        )  # hue_order=['', '']
    plt.xlabel('')
    # plt.xticks(rotation=90)
    plt.ylim((0.5, df.values.max()))
    if save:
        buffer = io.BytesIO()
        plt.savefig(buffer, format='pdf', bbox_inches='tight')
        buffer.seek(0)
    else:
        plt.show()
    plt.close()
    return buffer


def plot_weights_significant_correlations_to_buffer(
    weights_df, correlation_type, save=False
):
    """
    plot heatmap showing value of correlation if significant between different
    molecular signatures where weights_df.index.names is ['fold', 'class']
    """
    buffers_dict = {}
    correlation_cmap = sns.diverging_palette(
        240, 10, center='dark', as_cmap=True
    )  # Blue-Grey-Red
    # correlation all folds
    rho = weights_df.swaplevel('fold', 'class').sort_index().T.corr(
        method=correlation_type
    )
    sample_length = weights_df.shape[1]
    sig_mask = pd.DataFrame(
        significant_correlations(rho, sample_length)[0], dtype=np.uint8
    )
    # symmetric
    mask = np.zeros_like(rho, dtype=np.bool)
    mask[np.triu_indices(rho.shape[0], 1)] = True
    extended_mask = ~(sig_mask.astype(np.bool)).values | mask
    ticks = ['' for i in range(weights_df.shape[0])]
    ticks[0:weights_df.shape[0]:len(weights_df.index.levels[0])] = list(
        weights_df.index.levels[1]
    )
    f = sns.heatmap(
        rho,
        mask=extended_mask,
        vmin=-1,
        vmax=1,
        xticklabels=ticks,
        yticklabels=ticks,
        cmap=correlation_cmap
    ).get_figure()
    if save:
        buffers_dict['correlation_folds_{}'.format(correlation_type)
                     ] = io.BytesIO()
        f.savefig(
            buffers_dict['correlation_folds_{}'.format(correlation_type)],
            bbox_inches='tight'
        )
        buffers_dict['correlation_folds_{}'.format(correlation_type)].seek(0)
    else:
        plt.show()
    plt.close()

    # correlation of class summary
    median_classes_df = weights_df.swaplevel('fold',
                                             'class').sort_index().median(
                                                 axis=0, level='class'
                                             ).T
    median_classes_df['all folds'] = weights_df.median(
        axis=0
    ).T  # add median over folds over all classes
    rho_ = median_classes_df.corr(method=correlation_type)
    sig_mask_ = pd.DataFrame(
        significant_correlations(rho_, sample_length)[0],
        dtype=np.uint8,
        index=rho_.index,
        columns=rho_.columns
    )
    # symmetric
    mask_ = np.zeros_like(rho_, dtype=np.bool)
    mask_[np.triu_indices(rho_.shape[0], 1)] = True
    extended_mask_ = ~(sig_mask_.astype(np.bool)).values | mask_
    f = sns.heatmap(
        rho_, mask=extended_mask_, vmin=-1, vmax=1, cmap=correlation_cmap
    ).get_figure()
    if save:
        buffers_dict['correlation_phenotypes_{}'.format(correlation_type)
                     ] = io.BytesIO()
        f.savefig(
            buffers_dict['correlation_phenotypes_{}'.format(correlation_type)],
            bbox_inches='tight'
        )
        buffers_dict['correlation_phenotypes_{}'.format(correlation_type)
                     ].seek(0)
    else:
        plt.show()
    plt.close()
    return buffers_dict


def plot_weights_to_buffer(weights_df, save=False, plot_correlations=False):
    """
    plot molecular signature over many folds in pathway wise boxes.
    For non-binary problems each 1 versus Rest is ploted.
    weights_df.index.names should be ['fold'] or ['fold', 'class'])
    """
    buffers_dict = {}
    if 'class' not in weights_df.index.names:
        # _beautify_columns(weights_df)
        inducers_ordering = weights_df.median().sort_values(
            ascending=False
        ).index
        cols = [
            SIGNIFICANT_COLOR if sig else OTHER_COLOR
            for sig in significant_pathways(weights_df[inducers_ordering])
        ]
        avg = 1. / weights_df.shape[1]

        # plt.figure(figsize=(8,3))
        plt.figure(figsize=(18, 6))
        sns.boxplot(data=weights_df[inducers_ordering], palette=cols)
        plt.axhline(y=avg, lw=1., ls='--', c='black')
        # plt.title(
        #     'Pathways weights for {}\n'.format(phenotype_label) +
        #     '(fold={}, training samples per class={})'.format(cv, mc)
        # )
        plt.xlabel('Pathway')
        plt.ylabel('Weight')
        plt.ylim((0., weights_df.values.max()))
        # _ = plt.xticks([])
        _ = plt.xticks(rotation=90)
        if save:
            buffers_dict['weights_analysis'] = io.BytesIO()
            plt.savefig(
                buffers_dict['weights_analysis'],
                format='pdf',
                # '{}/weights_analysis_{}.pdf'.format(directory_name, suffix),
                bbox_inches='tight'
            )
            buffers_dict['weights_analysis'].seek(0)
        else:
            plt.show()
        plt.close()
    else:  # multi-index in case of EasyMKL with nonbinary labels
        significant_masks = {}
        # _beautify_columns(weights_df)
        w_dfs = {
            w: weights_df.xs(w, level='class')
            for w in weights_df.index.levels[1]
        }
        for w, w_df in w_dfs.items():
            inducers_ordering = w_df.median().sort_values(
                ascending=False
            ).index
            significant_masks[w] = significant_pathways(
                w_df[inducers_ordering]
            )
            cols = [
                SIGNIFICANT_COLOR if sig else OTHER_COLOR
                for sig in significant_masks[w]
            ]
            avg = 1. / w_df.shape[1]

            plt.figure(figsize=(18, 6))
            sns.boxplot(data=w_df[inducers_ordering], palette=cols)
            plt.axhline(y=avg, lw=1., ls='--', c='black')
            # plt.title(
            #     'Pathways weights for {} {} vs rest \n'.
            #     format(phenotype_label, w) +
            #     '(fold={}, training samples per class={})'.format(cv, mc)
            # )
            plt.xlabel('Pathway')
            plt.ylabel('Weight')
            plt.ylim((0., w_df.values.max()))
            # _ = plt.xticks([])
            _ = plt.xticks(rotation=90)
            if save:
                buffers_dict['weights_analysis_{}vR'.format(w)] = io.BytesIO()
                plt.savefig(
                    buffers_dict['weights_analysis_{}vR'.format(w)],
                    format='pdf',
                    bbox_inches='tight'
                )
                buffers_dict['weights_analysis_{}vR'.format(w)].seek(0)
            else:
                plt.show()
            plt.close()
        # 1vRest significance comparison
        significant_overall = pd.DataFrame(significant_masks).T
        # order = (  # sig count
        #     significant_overall.T
        # ).T.sum().sort_values().index[::-1]
        order = pd.concat([df for df in w_dfs.values()],
                          axis=0).median().sort_values(ascending=False).index

        fig, ax = plt.subplots()
        # plt.figure(figsize=(18,6))
        sns.heatmap(
            significant_overall[order],
            square=True,
            cmap=[OTHER_COLOR, SIGNIFICANT_COLOR],
            linewidths=.5,
            cbar=False,
            ax=ax
        )
        ax.set(aspect=5)
        plt.yticks(rotation=0)
        fontsize = 4 if w_df.shape[1] > 50 else 7
        plt.xticks(np.array(range(len(order))) + 0.5, order, fontsize=fontsize)
        plt.tight_layout()
        if save:
            buffers_dict['significant_pathways'] = io.BytesIO()
            plt.savefig(
                buffers_dict['significant_pathways'],
                format='pdf',
                bbox_inches='tight'
            )
            buffers_dict['significant_pathways'].seek(0)
        else:
            plt.show()
        plt.close()

        if plot_correlations:
            # correlation between 1vR weights
            more_buffers_dict = {}
            for correlation_type in ['pearson']:  # , 'spearman'
                more_buffers_dict = {
                    **more_buffers_dict,
                    **plot_weights_significant_correlations_to_buffer(
                        weights_df=weights_df,
                        correlation_type=correlation_type,
                        save=save
                    )
                }
            return {**buffers_dict, **more_buffers_dict}
    return buffers_dict
