import streamlit as st



class Resume:
  
    def display_information(self):
        st.header("About: ")
        st.write("**Current Organization**:Teleperformance Digital (Sep/2023- Working)| **Benori** (Oct/2021-Sep/2023) | **Strolar** \
                (Jan/2019-Sep/2020)")
        st.write( "A data science professional with over 3 years of experience, currently working at Teleperformance Digital. Specializing in Generative AI, machine learning, NLP, and data analytics using tools like Python, SQL, and Power BI. Strong background in developing predictive models, fraud detection systems, and customer service dashboards, with expertise in enhancing customer experience through advanced data science techniques.")
        st.write("")
        st.write("----------------------------")
        st.write("LinkedIn [link](https://www.linkedin.com/in/vipin-/)")
        st.write("GitHub [link](https://github.com/uservipin)")
        st.write("Web Project Source Code [link](https://github.com/uservipin/HF_Projects)")
        st.write("Data Science Projects [link](https://github.com/uservipin/data_science)")
        st.write("Python Projects [link](https://github.com/uservipin/python)")
   
    def display_work_experience(self):
        st.header("Work Experience")
        with st.expander("**Enhancing Customer Experience and Operational Efficiency through NLP Techniques**"):

            st.write('''
            This data science project aims to revolutionize call center operations by harnessing the power of NLP techniques. Through sentiment analysis, speech recognition, chatbots, and predictive analytics, the project seeks to improve customer experience, optimize resource allocation, and drive continuous improvement in service quality.\n
            **Sentiment Analysis**: Implement sentiment analysis algorithms to understand the emotional tone of customer calls, distinguishing between satisfaction, dissatisfaction, and anger. Additionally, identify the root causes behind these emotions.\n
            **Speech Recognition**: Develop a speech recognition system using AI that accurately transcribes spoken words into text, enabling further analysis such as sentiment analysis.\n
            **NLP-Powered Chatbots**: Integrate NLP capabilities into chatbots or virtual assistants to efficiently handle routine customer inquiries, providing prompt and personalized responses.\n
            **Insights**: Utilize NLP techniques to summarize call recordings, extract key insights, add relevant tags, and identify recurring customer complaints or trending inquiries.\n
            **Project Impact**:
            Enhanced Operational Efficiency: Automated processes such as call summarization, sentiment analysis, and predictive analytics reduce manual effort and streamline operations.

                        ''')
            
        with st.expander("**Credit Card Fraud Detection Risk Modelling**"):

            cr_pro_pointers = [
                            "Designed and implemented predictive models tailored for finance credit card fraud detection, leveraging expertise in data science",
                            "Specialized in developing churn, retention, and segmentation models to provide insights into customer behavior, enhancing customer engagement in the finance sector.",
                            "Model accuracy is 80% based on validation of alerts predicted by model",
                            "Utilized advanced statistical and machine learning techniques such asBoosting algorithms (XGboost, Ligt GBM, CAT Boosting) and complex datasets",
                            "Identified trends and patterns in customer behavior to effectively detect fraudulent activities and mitigate risks in credit card transactions",
                            "Contributed to market-based analysis by forecasting demand and implementing data-driven strategies to optimize business operations and minimize financial losses."
            ]

            bullet_points = f"<ul style='list-style-type: disc; padding-left: 15px;'>"  \
                    + "".join([f"<li>{item}</li>" for item in cr_pro_pointers]) + "</ul>"
            st.write(bullet_points, unsafe_allow_html=True)
                

        with st.expander("**Demand Forecast Model**"):

            cr_pro_pointers = [
                            "Developed and implemented predictive models for demand forecast market-based analysis, aiming to solve business challenges through data-driven insights.",
                            "Specialized in designing churn, retention, and segmentation models to provide companies with a deep understanding of customer behavior, thereby enhancing customer engagement and satisfaction",
                            "Leveraged various statistical and machine learning techniques such as Support Vector Machines (SVM), Voting, and ensemble algorithms to analyze extensive and intricate datasets.",
                            "Obtained data from diverse sources including Azure, ensuring data cleanliness and integrity through rigorous preprocessing and cleaning procedures.",
                            "Applied advanced ML models to identify trends and patterns in customer behavior, enabling proactive decision-making and strategic planning.",
                            "Contributed to market-based analysis by accurately forecasting demand and recommending actionable strategies to optimize business operations and drive revenue growth."
            ]

            bullet_points = f"<ul style='list-style-type: disc; padding-left: 15px;'>"  \
                    + "".join([f"<li>{item}</li>" for item in cr_pro_pointers]) + "</ul>"
            st.write(bullet_points, unsafe_allow_html=True)

        with st.expander("**Fraud Detection Analyst**"):

            cr_pro_pointers = [
                            "Proven track record in fraud detection within a call center environment, utilizing advanced analytical techniques and tools.",
                            "Implemented robust fraud detection algorithms like Random Forest and SVM and strategies, utilizing machine learning models and predictive analytics to identify suspicious activities and patterns in real time.",
                            "Conducted thorough investigations into fraudulent incidents, documenting findings and recommending preventative actions for future mitigation."
                        ]

            bullet_points = f"<ul style='list-style-type: disc; padding-left: 15px;'>"  \
                    + "".join([f"<li>{item}</li>" for item in cr_pro_pointers]) + "</ul>"
            st.write(bullet_points, unsafe_allow_html=True)
            
        with st.expander("**Power BI Dashboard with Analytics for Operational Insights :**"):

            cr_pro_pointers = [
                            "For the operation team, I created Power BI dashboards in the form of MIS operation reports. These reports encompassed metrics such as productivity, product quality, revenue, turnaround time (TAT), and Net Promoter Score (NPS). I employed DAX calculations and measures to derive key performance indicators (KPIs) including revenue, TAT, and NPS to achieve this. Additionally, I utilized DAX functions to create calculated columns and tables, enhancing the depth of data analysis.",
                            "I implemented row-level security functionality within Power BI to ensure data security and access control. This allowed for fine-grained control over data access based on user roles and permissions. Furthermore, I implemented incremental refresh techniques to optimize data loading and processing, ensuring efficient report performance and timely updates.",
                            "For seamless data integration, I leveraged the AWS-RDS MySQL database. This involved connecting Power BI to the database, and establishing a secure and reliable data connection. By utilizing the DAX logical operations, I provided robust calculation support, enabling accurate and precise analysis of the data."
                        ]

            bullet_points = f"<ul style='list-style-type: disc; padding-left: 15px;'>"  \
                    + "".join([f"<li>{item}</li>" for item in cr_pro_pointers]) + "</ul>"
            st.write(bullet_points, unsafe_allow_html=True)

      
    def skills_tools(self):
        st.header("Tools and Skills :")

        with st.expander("**Machine Learning**"):
            ML = ["SK-Learn",
                "Random Forest",
                "Decision Tree",
                "Ensemble Learning Bagging",
                "Boosting",
                "AUC/ROC",
                "EDA",
                "Clustering",
                "NLP",
                "Pipelines",
                "PCA" ]
            
            # Create bullet point list using HTML unordered list (<ul>) and list items (<li>)
            bullet_points = f"<ul style='list-style-type: disc; padding-left: 15px;'>"  \
                        + "".join([f"<li>{item}</li>" for item in ML]) + "</ul>"
            st.write(bullet_points, unsafe_allow_html=True)
                
        with st.expander("**Deep Learning**"):
            DL = ["BERT",
                "TensorFlow",
                "Transformers",
                "Encoders Decoders"]
            # Create bullet point list using HTML unordered list (<ul>) and list items (<li>)
            bullet_points = f"<ul style='list-style-type: disc; padding-left: 15px;'>"  \
                        + "".join([f"<li>{item}</li>" for item in DL]) + "</ul>"
            st.write(bullet_points, unsafe_allow_html=True)
            

        with st.expander("**Data Analytics**"):
            DA = ["Python (pandas, Scikit-Learn)",
                "SQL",
                "Power BI:(Power Query, Power Bi Service, DAX)"
                ]
            # Create bullet point list using HTML unordered list (<ul>) and list items (<li>)
            bullet_points = f"<ul style='list-style-type: disc; padding-left: 15px;'>"  \
                        + "".join([f"<li>{item}</li>" for item in DA]) + "</ul>"
            st.write(bullet_points, unsafe_allow_html=True)

        with st.expander("**Cloud**"):
            DA = ["- Data factory"
                "     - Basics",
                "     - ETL",
                "     - Pipeline",
                " - Azure Machine Learning Studio",
                "     - Train and test",
                "     - Data Preprocessing",
                "     - Run Experiments",
                "     - Auto ML",
                "     - Data Bricks (Basic)"
                ]
            st.markdown("\n".join(DA))


    def display_education_certificate(self):
        st.header("Education and Certificate:")

        with st.expander("**Certifications**"):
            Cer = ["Deep Learning – Andrew NG",
                "Data Analytics – Google",
                "SQL – Online",
                "Data Science/ Deep learning – CampusX",
                "Large Language Models with Semantic Search",
                "Preprocessing Unstructured Data for LLM Applications",
                "Building Applications with Vector Databases"

            ]
            # Create bullet point list using HTML unordered list (<ul>) and list items (<li>)
            bullet_points = f"<ul style='list-style-type: disc; padding-left: 15px;'>"  \
                        + "".join([f"<li>{item}</li>" for item in Cer]) + "</ul>"
            st.write(bullet_points, unsafe_allow_html=True)
                

        with st.expander("**Education**"):
            edu = ["ABES: B. Tech (ME) - 2019",
                    "Edyoda: Data Science -2021", 
                    "Coursera: Deep Learning -2022",
                    "Google: Data Analytics -2021"
                    ]
            # Create bullet point list using HTML unordered list (<ul>) and list items (<li>)
            bullet_points = f"<ul style='list-style-type: disc; padding-left: 15px;'>"  \
                        + "".join([f"<li>{item}</li>" for item in edu]) + "</ul>"
            st.write(bullet_points, unsafe_allow_html=True)

