import annotations.annotation_cleaner as annot_cleaner
import constants

from sklearn import metrics
from sklearn.decomposition import PCA
from scipy import stats as scipy_stat
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd
import numpy as np
import math
sns.set()

plot_label_dist=False
plot_kappa_dist=False
check_label_order=False
print_scores=False
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
    kappa = metrics.cohen_kappa_score(y1=annotation1, y2=annotation2, labels=kappa_label, weights="quadratic")
    if math.isnan(kappa) or kappa == 0.0:
        kappa = 0.0
    if kappa < 0.0:
        print(annotation1)
        print(annotation2)
    print("Current Kappa: " + str(kappa))
    return kappa

def get_correlation_score(annotation1, annotation2):
    # Pearson's Method
    correl, p_value = scipy_stat.pearsonr(annotation1, annotation2)
    if math.isnan(correl) or correl == 0.0:
        correl = 0.0
        print(annotation1)
        print(annotation2)
    print("Current Correl: " + str(correl))
    return correl

def get_interannotator_relaibility_score(annotation1, annotation2, method="correl"):
    agree_score = 0.0
    print("Inter-Annotator Score using " + method)
    if method == "correl":
        agree_score = get_correlation_score(annotation1, annotation2)
    elif method == "kappa":
        agree_score = get_kappa_score(annotation1, annotation2)
    return agree_score

def get_pairwise_agreeability_score(annotation_file, manifestation, annotators):
    pairwise_score={}
    annotator_wise_response =  annot_cleaner.get_annotator_wise_responses(annotation_file, manifestation, annotators)
    # Calculating Average ConvQ all datapoints
    final_average_score_all = {}
    for annotator in annotator_wise_response:
        final_average_score_all[annotator] = annotator_wise_response[annotator][manifestation + "_convq"].tolist()
    final_average_score_all = pd.DataFrame.from_dict(final_average_score_all).mean(axis = 1)
    for annotator1 in annotator_wise_response:
        for annotator2 in annotator_wise_response:
            if (annotator1 != annotator2) and (annotator2+"_"+annotator1 not in pairwise_score.keys()):
                convq_annotator1 =  annotator_wise_response[annotator1][manifestation+"_convq"].tolist()
                convq_annotator2 =  annotator_wise_response[annotator2][manifestation+"_convq"].tolist()
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
                    avg_convq = [(x + y)/2 for x, y in zip(convq_annotator1, convq_annotator2)]
                    sns.distplot(avg_convq, kde=False, rug=True).set_title("Final Average Annotations - " + manifestation)
                    plt.show()
                if check_label_order:
                    if manifestation == "group":
                        checker = annotator_wise_response[annotator1]["Group ID"].tolist() == annotator_wise_response[annotator2]["Group ID"].tolist()
                    else:
                        checker = annotator_wise_response[annotator1]["Individual ID"].tolist() == annotator_wise_response[annotator2]["Individual ID"].tolist()
                    print("ARE TWO LABELS IN SAME ORDER !!!!!!!!!!!!!! - " + str(checker))
                pairwise_score[annotator1+"_"+annotator2] = get_interannotator_relaibility_score(convq_annotator1,
                                                                                                 convq_annotator2, method="correl")
    return pairwise_score, final_average_score_all


def get_final_agreeability_score_for(annotation_file=constants.group_conq_annot_data, manifestation="group", annotators=["Nakul", "Swathi", "Divya"]):
    pairwise_score, final_average_score_all = get_pairwise_agreeability_score(annotation_file, manifestation, annotators)
    score=0
    print("Pairwise SCORES - " + str(pairwise_score))
    for pairs in pairwise_score.keys():
        score = score + pairwise_score[pairs]
    score = score/len(pairwise_score.keys())
    return score, final_average_score_all

def calculate_groupwise_agreeability_score(annotation_file=constants.group_conq_annot_data, manifestation="group", annotators=["Nakul", "Swathi", "Divya"]):
    pairwise_score={}
    pairwise_groupwise_score={}
    annotator_wise_response =  annot_cleaner.get_groupwise_annotator_responses(annotation_file, manifestation, annotators)

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
                if plot_kappa_dist:
                    sns.distplot(groups_score, kde=False, rug=True).set_title(measure + " Score Dist - All groups: For - " + annotator1+"_"+annotator2)
                    plt.show()
                if check_label_order:
                    if manifestation == "group":
                        checker = annotator_wise_response[annotator1]["Group ID"].tolist() == \
                                  annotator_wise_response[annotator2]["Group ID"].tolist()
                    else:
                        checker = annotator_wise_response[annotator1]["Individual ID"].tolist() == \
                                  annotator_wise_response[annotator2]["Individual ID"].tolist()
                    print("ARE TWO LABELS IN SAME ORDER !!!!!!!!!!!!!! - " + str(checker))
                # groups_score = pd.DataFrame(groups_score).dropna()[0].values
                pairwise_score[annotator1+"_"+annotator2] = mean(groups_score)
                pairwise_groupwise_score[annotator1+"_"+annotator2] = groups_score
    return pairwise_score, pairwise_groupwise_score

def get_final_groupwise_agreeability_score_for(annotation_file=constants.group_conq_annot_data, manifestation="group", annotators=["Nakul", "Swathi", "Divya"]):
    pairwise_score, pairwise_groupwise_score = calculate_groupwise_agreeability_score(annotation_file=annotation_file, manifestation=manifestation, annotators=annotators)
    score=0
    print("Pairwise SCORES - " + str(pairwise_score))
    for pairs in pairwise_score.keys():
        score = score + pairwise_score[pairs]
    score = score/len(pairwise_score.keys())
    return score, pairwise_groupwise_score

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
    cleaned_annotations = annot_cleaner.derive_convq_scores_for_reponses(annotation_file, manifestation, calculate_convq=False, reverse_scale=False)
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


manifest="group"
file=constants.group_conq_annot_data
measure="correl"

if plot_pca:
    perform_pca_analysis_on(file, manifest, ["Nakul", "Swathi", "Divya"])
else:
    if sum_wise:
        score, final_average_score_all = get_final_agreeability_score_for(annotation_file=file, manifestation=manifest, annotators=["Nakul", "Swathi", "Divya"])
        print("(Calculated Overall) Mean Pair-wise AGREEABILITY - " + str(score))
        if plot_final_convq:
            sns.distplot(final_average_score_all, kde=False, rug=True).set_title("MEAN ConvQ - " + manifest)
            plt.show()

    if group_wise:
        score, pairwise_groupwise_score = get_final_groupwise_agreeability_score_for(annotation_file=file,
                                                                                          manifestation=manifest,
                                                                                          annotators=["Nakul", "Swathi",
                                                                                                      "Divya"])
        print(pairwise_groupwise_score)
        pairwise_groupwise_score_meaned = pd.DataFrame.from_dict(pairwise_groupwise_score).mean(axis = 1)
        print("(Calculated Group-Wise) Mean Pair-wise AGREEABILITY - " + str(score))
        if plot_final_convq:
            sns.distplot(pairwise_groupwise_score_meaned, bins=10, kde=False, rug=True).set_title("Mean AGREEABILITY SCORE - " + manifest)
            plt.show()

    if mean_kappa_vs_conq_plot:
        plt.scatter(pairwise_groupwise_score_meaned, final_average_score_all,s=8,c='b')
        plt.ylabel('Mean ConvQ Score - ' + manifest)
        plt.xlabel('Mean Kappa Score')
        plt.show()