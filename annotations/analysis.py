import annotations.annotation_cleaner as annot_cleaner
import constants

from sklearn import metrics
from scipy import stats as scipy_stat
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd

plot_label_dist=False
plot_kappa_dist=False
check_label_order=True
print_scores=False
plot_final_convq=True

def get_kappa_score(annotation1, annotation2):
    # Cohen's Method
    kappa = metrics.cohen_kappa_score(y1=annotation1, y2=annotation2, weights="linear")
    return kappa

def get_correlation_score(annotation1, annotation2):
    # Pearson's Method
    correl, p_value = scipy_stat.pearsonr(annotation1, annotation2)
    print("Correl p-value: " + str(p_value))
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
                        group_score = get_interannotator_relaibility_score(group_response1, group_response2, "kappa")
                        groups_score.append(group_score)
                else:
                    for group_id, indiv_id in zip(annotator_wise_response[annotator1]["Group ID"].tolist(), annotator_wise_response[annotator1]["Individual ID"].tolist()):
                        indiv_response1 = annotator_wise_response[annotator1].loc[
                            (annotator_wise_response[annotator1]['Group ID'] == str(group_id)) &
                            (annotator_wise_response[annotator1]["Individual ID"] == indiv_id)][col_list].values[0]
                        indiv_response2 = annotator_wise_response[annotator2].loc[
                            (annotator_wise_response[annotator2]['Group ID'] == str(group_id)) &
                            (annotator_wise_response[annotator2]["Individual ID"] == indiv_id)][col_list].values[0]
                        group_score = get_interannotator_relaibility_score(indiv_response1, indiv_response2, "kappa")

                        if group_score ==0 and 0:
                            print(indiv_response1)
                            print(indiv_response2)

                        groups_score.append(group_score)
                if plot_kappa_dist:
                    sns.distplot(groups_score, kde=False, rug=True).set_title("Kappa Score Dist - All groups")
                    plt.show()
                if check_label_order:
                    if manifestation == "group":
                        checker = annotator_wise_response[annotator1]["Group ID"].tolist() == \
                                  annotator_wise_response[annotator2]["Group ID"].tolist()
                    else:
                        checker = annotator_wise_response[annotator1]["Individual ID"].tolist() == \
                                  annotator_wise_response[annotator2]["Individual ID"].tolist()
                    print("ARE TWO LABELS IN SAME ORDER !!!!!!!!!!!!!! - " + str(checker))
                groups_score = pd.DataFrame(groups_score).dropna()[0].values
                pairwise_score[annotator1+"_"+annotator2] = mean(groups_score)
    return pairwise_score

manifest="group"
score, final_average_score_all = get_final_agreeability_score_for(annotation_file=constants.group_conq_annot_data, manifestation=manifest, annotators=["Nakul", "Swathi", "Divya"])
print("FINAL AGREEABILITY SCORE = " + str(score))
if plot_final_convq:
    sns.distplot(final_average_score_all, kde=False, rug=True).set_title("Final Average Annotations - " + manifest)
    plt.show()
# print(calculate_groupwise_agreeability_score(annotation_file=constants.group_conq_annot_data, manifestation="group", annotators=["Nakul", "Swathi", "Divya"]))