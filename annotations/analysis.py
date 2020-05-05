import annotations.annotation_cleaner as annot_cleaner
import constants

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy import stats as scipy_stat
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
import pingouin as pg
import pandas as pd
import numpy as np
import math
sns.set()

check_label_order=False
print_scores=False

plot_label_dist=False
plot_kappa_dist=False
plot_final_convq=False
plot_pca=False

group_wise=True
sum_wise=True

mean_kappa_vs_conq_plot=True

if group_wise:
    kappa_label = list(range(1,6))
else:
    kappa_label = list(range(1,51))

def get_kappa_score(annotation1, annotation2):
    # Cohen's Method
    # print(kappa_label)
    # annotation1=np.around(annotation1, decimals=1)
    # annotation2=np.around(annotation2, decimals=1)
    # kappa_label = np.unique(np.array(list(annotation1) + list(annotation2)))
    # kappa_label.extend(annotation2)
    # kappa_label=np.unique(kappa_label)#.sort()
    # kappa_label.sort()
    # kappa_label = np.arange(-3,4)
    # print(annotation1)
    # print(annotation2)
    # print(kappa_label)
    kappa = metrics.cohen_kappa_score(y1=annotation1, y2=annotation2, weights="quadratic")
    if math.isnan(kappa) or kappa == 0.0:
        kappa = 0.0
    # if kappa < 0.0:
    #     print(annotation1)
    #     print(annotation2)
    # print("Current Kappa: " + str(kappa))
    return kappa

# Taken From - https://github.com/statsmodels/statsmodels/issues/1272
def get_cronbach_alpha(itemscores):
    # cols are items, rows are observations
    # itemscores = np.asarray(itemscores)
    # print(itemscores)
    # itemvars = itemscores.var(axis=0, ddof=1)
    # tscores = itemscores.sum(axis=1)
    # nitems = itemscores.shape[1]
    # calpha = nitems / float(nitems-1) * (1 - itemvars.sum() / float(tscores.var(ddof=1)))
    calpha = pg.cronbach_alpha(data=pd.DataFrame(itemscores))[0]
    print("Current Alpha: " + str(calpha))
    return calpha

def get_correlation_score(annotation1, annotation2):
    # Pearson's Method
    correl, p_value = scipy_stat.pearsonr(annotation1, annotation2)
    if math.isnan(correl) or correl == 0.0:
        correl = 0.0
    # if correl < 0.0:
    #     print(annotation1)
    #     print(annotation2)
    # print("Current Correl: " + str(correl))
    return correl

def get_spearman_correlation_score(annotation1, annotation2):
    # Pearson's Method
    correl, p_value = scipy_stat.spearmanr(annotation1, annotation2)
    if math.isnan(correl) or correl == 0.0:
        correl = 0.0
    # if correl < 0.0:
    #     print(annotation1)
    #     print(annotation2)
    # print("Current Correl: " + str(correl))
    return correl

def get_mse_score(annotation1, annotation2):
    # MSE
    mse = mean_squared_error(annotation1, annotation2)
    return mse

def get_interannotator_relaibility_score(annotation1, annotation2, method="correl"):
    agree_score = 0.0
    # print("Inter-Annotator Score using " + method)
    if method == "correl":
        agree_score = get_correlation_score(annotation1, annotation2)
    elif method == "kappa":
        agree_score = get_kappa_score(annotation1, annotation2)
    elif method == "alpha": # cronbach alpha
        agree_score = get_cronbach_alpha(np.vstack((annotation1, annotation2)))
    elif method == "spearman":
        agree_score = get_spearman_correlation_score(annotation1, annotation2)
    elif method == "mse":
        agree_score = mean_squared_error(annotation1, annotation2)
    return agree_score

def agree_score_calculator_debuggers(annotator_wise_response, annotator1, annotator2, manifestation, groups_score):
    if plot_kappa_dist:
        sns.distplot(groups_score, kde=False, rug=True).set_title(
            measure + " Score Dist - All groups: For - " + annotator1 + "_" + annotator2)
        plt.show()
    if check_label_order:
        if manifestation == "group":
            checker = annotator_wise_response[annotator1]["Group ID"].tolist() == \
                      annotator_wise_response[annotator2]["Group ID"].tolist()
        else:
            checker = annotator_wise_response[annotator1]["Individual ID"].tolist() == \
                      annotator_wise_response[annotator2]["Individual ID"].tolist()
        print("ARE TWO LABELS IN SAME ORDER !!!!!!!!!!!!!! - " + str(checker))

    return True

def convq_calculator_debuggers(annotator_wise_response, annotator1, annotator2, manifestation, convq_annotator1, convq_annotator2):
    if print_scores:
        if manifestation == "group":
            print(annotator_wise_response[annotator1]["Group ID"].tolist())
        else:
            print(annotator_wise_response[annotator1]["Individual ID"].tolist())
        print(convq_annotator1)
        print(convq_annotator2)

    if plot_label_dist:
        sns.distplot(convq_annotator1, kde=False, rug=True).set_title("Annotator 1")
        plt.show()
        sns.distplot(convq_annotator2, kde=False, rug=True).set_title("Annotator 2")
        plt.show()
        avg_convq = [(x + y) / 2 for x, y in zip(convq_annotator1, convq_annotator2)]
        sns.distplot(avg_convq, kde=False, rug=True).set_title("Final Average Annotations - " + manifestation)
        plt.show()
    if check_label_order:
        if manifestation == "group":
            checker = annotator_wise_response[annotator1]["Group ID"].tolist() == annotator_wise_response[annotator2][
                "Group ID"].tolist()
        else:
            checker = annotator_wise_response[annotator1]["Individual ID"].tolist() == \
                      annotator_wise_response[annotator2]["Individual ID"].tolist()
        print("ARE TWO LABELS IN SAME ORDER !!!!!!!!!!!!!! - " + str(checker))
    return True

def get_pairwise_agreeability_score(annotation_file, manifestation, annotators, zero_mean=False):

    annotator_wise_response =  annot_cleaner.get_annotator_wise_responses(annotation_file, manifestation, annotators, zero_mean)
    for annotator in annotator_wise_response.keys():
        if manifestation == "group":
            groups_label = annotator_wise_response[annotator]["Group ID"].tolist()
        else:
            grp_label = annotator_wise_response[annotator]["Group ID"].tolist()
            ind_label = annotator_wise_response[annotator]["Individual ID"].tolist()
            groups_label = [str(a_) + "_" + str(b_) for a_, b_ in zip(grp_label, ind_label)]
        break
    # Calculating Average ConvQ all datapoints
    final_average_score_all = {}
    for annotator in annotator_wise_response:
        final_average_score_all[annotator] = annotator_wise_response[annotator][manifestation + "_convq"].tolist()
    final_average_score_all = pd.DataFrame.from_dict(final_average_score_all).mean(axis = 1)

    # for annotator1 in annotator_wise_response:
    #     for annotator2 in annotator_wise_response:
    #         if (annotator1 != annotator2) and (annotator2+"_"+annotator1 not in pairwise_score.keys()):
    #             convq_annotator1 =  annotator_wise_response[annotator1][manifestation+"_convq"].tolist()
    #             convq_annotator2 =  annotator_wise_response[annotator2][manifestation+"_convq"].tolist()
    #             # Just a debugger func., No Logic in it
    #             score_calculator_debuggers(annotator_wise_response, annotator1, annotator2, manifestation, convq_annotator1, convq_annotator2)
    #             pairwise_score[annotator1+"_"+annotator2] = get_interannotator_relaibility_score(convq_annotator1,
    #                                                                                              convq_annotator2, method=measure)
    return final_average_score_all, groups_label


def get_final_convq_score_for(annotation_file=constants.group_conq_annot_data, manifestation="group", annotators=["Nakul", "Swathi", "Divya"], zero_mean=False):
    # Get Conversation Quality Scores
    print("HEREREERERERERRERer")
    final_average_convq, groups_label_1 = get_pairwise_agreeability_score(annotation_file, manifestation, annotators, False)
    # Get Inter-Rater Agreeability for ConvQ scores
    pairwise_kappa_score, final_average_kappa, groups_label_2 = get_final_groupwise_agreeability_score_for(annotation_file, manifestation, annotators, zero_mean)
    #Conversion from pairwise kappa to final
    final_average_kappa = pd.DataFrame.from_dict(final_average_kappa).mean(axis=1).values.tolist()
    return np.mean(final_average_convq.values), pairwise_kappa_score, final_average_convq.values.tolist(), final_average_kappa, groups_label_1 # groups_label_1 and groups_label_2 are same -> Coming from different datasources (Values Checked)

def get_stacked_annotator_respones(annotator_wise_response , col_list, group_id, indiv_id, manifest):
    stacked_data=[]
    for annotator in annotator_wise_response.keys():
        if manifest == "group":
            annotator_data = \
            annotator_wise_response[annotator].loc[annotator_wise_response[annotator]['Group ID'] == str(group_id)][
                col_list].values[0]
        else:
            annotator_data = annotator_wise_response[annotator].loc[
                (annotator_wise_response[annotator]['Group ID'] == str(group_id)) &
                (annotator_wise_response[annotator]["Individual ID"] == indiv_id)][col_list].values[0]
        if len(stacked_data) == 0:
            stacked_data = annotator_data
        else:
            stacked_data = np.vstack((stacked_data, annotator_data))
    return stacked_data


def calculate_groupwise_agreeability_score(annotation_file=constants.group_conq_annot_data, manifestation="group", annotators=["Nakul", "Swathi", "Divya"], zero_mean=False):
    pairwise_score={}
    pairwise_groupwise_score={}

    annotator_wise_response =  annot_cleaner.get_groupwise_annotator_responses(annotation_file, manifestation, annotators, zero_mean=zero_mean)
    # Just to create Groubs order list for checks
    for annotator in annotator_wise_response.keys():
        if manifestation == "group":
            groups_label = annotator_wise_response[annotator]["Group ID"].tolist()
        else:
            grp_label = annotator_wise_response[annotator]["Group ID"].tolist()
            ind_label = annotator_wise_response[annotator]["Individual ID"].tolist()
            groups_label = [str(a_) +  "-" + str(b_) for a_, b_ in zip(grp_label, ind_label)]
        break

    #ZERO MEAN BEFORE CALCUALTION of KAPPA Scores below
    # Now Doing Cronbach's alpha - Below code under IF, only for Cronbach calc - no pairwise
    if False:#measure == "alpha":
        print("DOING Cronbach's alpha............")
        groups_score = []
        col_list = list(annotator_wise_response[list(annotator_wise_response.keys())[0]])
        col_list.remove("Group ID")
        key = "-".join(list(annotator_wise_response.keys())[0])
        if manifestation == "group":
            for group_id in annotator_wise_response[list(annotator_wise_response.keys())[0]]["Group ID"].tolist():
                group_annotation_stack = get_stacked_annotator_respones(annotator_wise_response , col_list, group_id, "", manifest)
                score = get_cronbach_alpha(group_annotation_stack)
                groups_score.append(score)
        else:
            col_list.remove("Individual ID")
            for group_id, indiv_id in zip(annotator_wise_response[list(annotator_wise_response.keys())[0]]["Group ID"].tolist(),
                                          annotator_wise_response[list(annotator_wise_response.keys())[0]]["Individual ID"].tolist()):
                indiv_annotation_stack = get_stacked_annotator_respones(annotator_wise_response , col_list, group_id, indiv_id, manifest)
                score = get_cronbach_alpha(indiv_annotation_stack)
                groups_score.append(score)

        pairwise_score[key] = mean(groups_score)
        pairwise_groupwise_score[key] = groups_score
    else:
        for annotator1 in annotator_wise_response:
            for annotator2 in annotator_wise_response:
                if (annotator1 != annotator2) and (annotator2 + "_" + annotator1 not in pairwise_score.keys()):
                    groups_score=[]
                    col_list = list(annotator_wise_response[annotator1])
                    col_list.remove("Group ID")
                    if manifestation == "indiv":
                        col_list.remove("Individual ID")

                    if manifestation == "group":
                        for group_id in annotator_wise_response[annotator1]["Group ID"].tolist():
                            group_response1 = annotator_wise_response[annotator1].loc[annotator_wise_response[annotator1]['Group ID'] == str(group_id)][col_list].values[0]
                            group_response2 = annotator_wise_response[annotator2].loc[annotator_wise_response[annotator2]['Group ID'] == str(group_id)][col_list].values[0]
                            group_score = get_interannotator_relaibility_score(group_response1, group_response2, measure)
                            groups_score.append(group_score)
                    else:
                        for group_id, indiv_id in zip(annotator_wise_response[annotator1]["Group ID"].tolist(), annotator_wise_response[annotator1]["Individual ID"].tolist()):
                            indiv_response1 = annotator_wise_response[annotator1].loc[
                                (annotator_wise_response[annotator1]['Group ID'] == str(group_id)) &
                                (annotator_wise_response[annotator1]["Individual ID"] == indiv_id)][col_list].values[0]
                            indiv_response2 = annotator_wise_response[annotator2].loc[
                                (annotator_wise_response[annotator2]['Group ID'] == str(group_id)) &
                                (annotator_wise_response[annotator2]["Individual ID"] == indiv_id)][col_list].values[0]
                            group_score = get_interannotator_relaibility_score(indiv_response1, indiv_response2, measure)
                            groups_score.append(group_score)
                    #Debugger Func.,
                    agree_score_calculator_debuggers(annotator_wise_response, annotator1, annotator2, manifestation, groups_score)
                    pairwise_score[annotator1+"_"+annotator2] = mean(groups_score)
                    pairwise_groupwise_score[annotator1+"_"+annotator2] = groups_score

    return pairwise_score, pairwise_groupwise_score, groups_label

def get_final_groupwise_agreeability_score_for(annotation_file=constants.group_conq_annot_data, manifestation="group", annotators=["Nakul", "Swathi", "Divya"], zero_mean=False):
    pairwise_score, pairwise_groupwise_score, groups_label = calculate_groupwise_agreeability_score(annotation_file=annotation_file,
                                                                                                    manifestation=manifestation, annotators=annotators,
                                                                                                    zero_mean=zero_mean)
    score=0
    # print("Pairwise SCORES - " + str(pairwise_score))
    for pairs in pairwise_score.keys():
        score = score + pairwise_score[pairs]
    score = score/len(pairwise_score.keys())
    return score, pairwise_groupwise_score, groups_label

def plot_pca_loadings(score, coeff, labels=None):
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley,s=4,c='grey', alpha=0.3)
    plt.grid(True, which='both')
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='black', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, str(i + 1), color='black', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='black', ha='center', va='center')

def plot_eigenvalue_bar_graph(eigen_values):
    print(sum(eigen_values))
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Componenets')
    plt.title('PCA Analysis')
    var = np.cumsum(np.round(eigen_values, decimals=3) * 10)
    plt.plot(var)
    plt.bar(range(len(eigen_values)),eigen_values*10)

def perform_pca_analysis_on(annotation_file=constants.group_conq_annot_data, manifestation="group", annotators=["Nakul", "Swathi", "Divya"]):
    cleaned_annotations = annot_cleaner.derive_convq_scores_for_reponses(annotators, annotation_file, manifestation, calculate_convq=False, reverse_scale=False, zero_mean=False)
    col_list = list(cleaned_annotations)
    if manifestation == "group":
        column_omit = ["Annotator Name", "Group ID"]
    else:
        column_omit = ["Annotator Name", "Group ID", "Individual ID"]
    for col in column_omit:
        col_list.remove(col)

    pca = PCA()
    x_trans = pca.fit_transform(cleaned_annotations[col_list])
    # Call the function. Use only the 2 PCs.
    plot_pca_loadings(x_trans[:, 0:2], np.transpose(pca.components_[0:2, :]))
    plt.show()
    plot_eigenvalue_bar_graph(pca.explained_variance_)
    plt.show()

# IMPORTANT FOR MODELLING
measure="kappa"

if False:
    manifest="indiv"
    file=constants.indiv_conq_annot_data
    annotators=["Nakul", "Divya"]#, "Swathi"]
    zero_mean=True
    # "Nakul", "Divya", "Swathi"

    if plot_pca:
        perform_pca_analysis_on(file, manifest, annotators)
    else:
        # if sum_wise:
        score_convq, score_kappa, final_average_convq, final_average_kappa, groups_label = get_final_convq_score_for(annotation_file=file,
                                                                                                                     manifestation=manifest,
                                                                                                                     annotators=annotators,
                                                                                                                     zero_mean=zero_mean)
        # print(groups_label)
        print("(Calculated Overall) Mean Pair-wise CONVQ - " + str(score_convq))
        if plot_final_convq:
            sns.distplot(final_average_convq, kde=False, rug=True).set_title("MEAN ConvQ - " + manifest)
            plt.show()

        pairwise_groupwise_score_meaned = pd.DataFrame.from_dict(final_average_kappa).mean(axis = 1)
        print("(Calculated Group-Wise) Mean Pair-wise AGREEABILITY - " + str(score_kappa))
        if plot_final_convq:
            sns.distplot(pairwise_groupwise_score_meaned, bins=10, kde=False, rug=True).set_title("Mean AGREEABILITY SCORE - " + manifest)
            plt.show()

        if mean_kappa_vs_conq_plot:
            filtered_ind = np.where(np.array(pairwise_groupwise_score_meaned) >= -10)
            pairwise_groupwise_score_meaned = np.array(pairwise_groupwise_score_meaned)[filtered_ind[0]].tolist()
            final_average_score_all = np.array(final_average_convq)[filtered_ind[0]].tolist()
            plt.scatter(pairwise_groupwise_score_meaned, final_average_score_all,s=8,c='b')
            if False:
                for i, txt in enumerate(groups_label):
                    plt.annotate(txt, (pairwise_groupwise_score_meaned[i],
                                       final_average_score_all[i]))

            plt.ylabel('Mean ConvQ Score - ' + manifest)
            plt.xlabel('Mean Kappa Score')
            plt.show()