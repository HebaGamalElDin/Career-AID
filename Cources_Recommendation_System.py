# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:20:23 2020

@author: Heba Gamal EL-Din
"""

import pandas as pd
import ast
from textblob import Word
from itertools import chain
from gensim.models import FastText
from TextSummarization import Text_Summarization
import string

class Backend:
    def __init__(self):
        ########################
        """ Courses Dataset """
        #######################
        self.DF1 = pd.read_excel("courses.xlsx", 'in', index_col = 0).drop_duplicates().drop(['_id','ratings_count', 'duration', 'category', 'level', 'schoolName', 'instructors', 'enrolled_students_count', 'num_reviews'], axis=1)
        self.DF1 = self.DF1[(self.DF1.title != "NaN") & (self.DF1.skills != "NaN")]
        
        """                            PreProcessing                               """
        self.DF1.skills = self.DF1.skills.str.replace(r'[^\w\s,[ ] ]' , '').str.lower()
        self.DF1.skills = self.DF1.skills.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        self.DF1.skills = self.DF1.skills.apply(ast.literal_eval)
        
        ##########################
        """ Job Titles Dataset """
        ##########################
        self.DF2 = pd.read_csv("jobs_skills.csv").drop(['_id', 'industry'], axis =1).drop_duplicates(subset='title').reset_index().drop(['index'], axis=1)
        self.DF2 = self.DF2[(self.DF2.title != "NaN") & (self.DF2.skills != "NaN")]
        
        """                            Text Vectorization                          """
        self.Model = FastText(self.DF1.skills, min_count = 1, size = 40, window = 5)
        self.Model2 = FastText(self.DF2.skills, min_count = 1, size = 40, window = 5)

    

        """                               PreProcessing                            """
        def clean_job_titles_list(job_titles_list):
            all_job_titles = [ str(cleaned_title) for cleaned_title in job_titles_list ]
            all_job_titles = [ title.split(' - ')[0].strip().lower().split(',')[0].split('@')[0].strip().split(" at ")[0].strip().split('\\u')[0] for title in all_job_titles  if "-end" != title.lower()]
            all_job_titles = [ title.encode('ascii', 'ignore').decode("utf-8").replace("[fcis19]",'').split(" for ")[0].split(" in ")[0].strip() for title in all_job_titles ]
            all_job_titles = [ title.split("(")[0].strip() for title in all_job_titles]
            cleaned_job_titles = []
            for title in all_job_titles:
                if len(title.split("|")) >= 2:
                    cleaned_job_titles.append(title.split("|")[0].strip() + " | " + title.split("|")[1].strip())
                else: 
                    cleaned_job_titles.append(title.split("/")[0].strip())
            return cleaned_job_titles
        
        self.DF2.title =  clean_job_titles_list(self.DF2.title.values)
        self.DF2.skills = self.DF2.skills.str.replace(r'[^\w\s,[ ] ]' , '').str.lower()
        self.DF2.skills = self.DF2.skills.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        self.DF2.skills = self.DF2.skills.apply(ast.literal_eval)
    
    """                            Backend Functions                           """
    def  Jobs_2_Skills_(self,jobTitle):
        Scores_ = {}
        Job_title = str(jobTitle)
        for title in enumerate(self.DF2.title.values):
            if title[1]:
                Scores_[title[0]] = self.Model2.wv.n_similarity(title[1], Job_title)
        Scores_Final = sorted(Scores_.items() , key=lambda x: x[1], reverse=True)[:5]
        Indecies = [index[0] for index in Scores_Final]
        Skills = self.DF2.skills.iloc[Indecies]
        Functions = self.DF2.jobFunction.iloc[Indecies]
        return Indecies, Skills, Functions
    
    def  Skills_2_Courses_(self, Skills_Set):
        Scores_ = {}
        if type(Skills_Set) is str:
            Skills_Set= [inp.lower() for inp in Skills_Set.split()]
        for skill in enumerate(self.DF1.skills.values):
            if skill[1]:
                Scores_[skill[0]] = self.Model.wv.n_similarity(skill[1], Skills_Set)
        Scores_Final = sorted(Scores_.items() , key=lambda x: x[1], reverse=True)[:5]
        Indecies = [index[0] for index in Scores_Final]
        Courses = self.DF1.iloc[Indecies].drop_duplicates(subset='title')
        return Indecies, Courses

    """ Course Description Summarization """
    
    def Pipeline_(self, Title):
        JobTitle = str(Title)
        indxx,Skills, Functions = self.Jobs_2_Skills_(JobTitle)
        Skills_Set = list(set(chain(*Skills.tolist())))
        table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        Functions = [function.translate(table) for function in Functions]
        Functions_ = list(set(chain(*[Functions])))
        Skills_Set = [skill.title().translate(table) for skill in Skills_Set]
        print("\n[INFO] Recommendatios Are Being Loaded ...\n")
        Indeces, Courses = self.Skills_2_Courses_(Skills_Set)
        courses = []
        for indx in Indeces:
            Desc = str(self.DF1.iloc[indx][2])
            if len(Desc) > 200:
                Model = Text_Summarization(self.DF1.iloc[indx][2])
                Desc = Model.summarize_()
            courses.append((self.DF1.iloc[indx][0], self.DF1.iloc[indx][1], self.DF1.iloc[indx][-1], Desc))
        return Skills_Set, courses, Functions_

#######################
""" Main Function """ #
#######################
if __name__ == "__main__":   
    clas = Backend()
    JOB = input("[INFO] Enter Job Title Here.. => ")
    Skills, Courses, Functions = clas.Pipeline_(JOB)    
    print("To Become A %s Follow The Upcoming Recommendations\n"%JOB)
    print(" "*(len("Skills")//2)+"Skills\n"+"="*len("Skills")*2)
    for skill in Skills:
        print(skill)
    print("\n\n")
    print(" "*(len("Job Functions")//2)+"Job Functions\n"+"="*len("Job Functions")*2)
    for function in Functions:
        print(function)    
    print("\n\n")  
    print(" "*(len("Courses")//2)+"Courses \n"+"="*len("Courses")*2)
    for indx,course in enumerate(Courses):
        print(str(indx+1) + ") Course Name => " + course[0] + "\n")
        print("Course URL => " + course[1] + "\n")
        print("Source => " + course[2] + "\n")
        print("Course Description =>\n " + course[3] + "\n")

