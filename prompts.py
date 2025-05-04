from pydantic import BaseModel, Field
from textwrap import dedent

class AtomicClaims(BaseModel):
    atomic_claims: list[str] = Field(description="A list of atomic claims extracted from the text.")

class RedundantClaimIndices(BaseModel):
    redundant_claim_indices: list[int] = Field(description="indices of the redundant claims.")

interrogator_prompts = {
#     "q_from_claims_system_prompt": """You are a helpful assistant.""",
#     "q_from_claims_user_prompt": """You are given a context and a list of claims. Generate 1-3 most likely question(s) that can be answered by the claims.

# <Context>{context}</Context>
# <Claims>
# {claims}
# </Claims>""",
    "extract_ac_system_prompt": """You are a helpful assistant.""",
    "extract_ac_user_prompt": """Given context and a paragraph of text, deconstruct the text into the smallest possible standalone and self-contained facts without semantic repetition. Each fact should come from the text and must be related to the context.

<Context>{context}</Context>
<Text>{text}</Text>""",

    "q_from_claims_system_prompt": """You are a helpful assistant.""",
    "q_from_claims_user_prompt": """Given context and a list of claims, generate specific, clear questions that have their answers contained in the corresponding claims.
For each claim, generate 1-3 questions that ask for factual information. The generated questions should start from asking the most general information and must be related to the context.

<Context>{context}</Context>
<Claims>
{claims}
</Claims>""",

    "q_from_single_claim_system_prompt": """You are a helpful assistant.""",
    "q_from_single_claim_user_prompt": """Given context and a claim, generate one specific, clear question that has its answer contained in the claim. The generated question must be self-contained and related to the context.
Return only the question, with no additional text.

Context: {context}

Claim: {claim}""",

    "rank_q_system_prompt": """You are a helpful assistant.""",
    "rank_q_user_prompt": """You are given a list of numbered questions. Carefully follow the instructions below to compile the questions into ranked question sets:
1. Cluster similar questions into sets, where each pair of questions in a set are similar to each other. Two questions are similar if they are phrased similarly, or potentially have the same answer.
2. Rerank the question sets by generality, that is, the most general questions should be ranked first, and the most specific, focused questions should be ranked last.
3. Return a list of ranked question sets, where each set is a list of question numbers given in the original question list.

<Question List>
{question_list}
</Question List>""",

    "rm_redundant_ac_system_prompt": """You are a helpful assistant who carefully examines difference between statements.""",
    "rm_redundant_ac_user_prompt_2": """Given two lists of claims, check the redundancy for claims in list B with respect to claims in list A.
For each claim in list B, if it contains information that is already present in any claim in list A, label it as a redundant claim.

<Claim List A>
{claim_list_A}
</Claim List A>

<Claim List B>
{claim_list_B}
</Claim List B>""",

    "rm_redundant_ac_user_prompt": """Given two lists of claims, check the redundancy for claims in list B with respect to claims in list A.
Claim in list B is deemed redundant if it contains information that is already present in any claim in list A. Return a list of indices of the redundant claims in list B.

<Claim List A>
{claim_list_A}
</Claim List A>

<Claim List B>
{claim_list_B}
</Claim List B>""",

    "cluster_similar_ac_system_prompt": """You are a helpful assistant.""",
    "cluster_similar_ac_user_prompt": """You are given a list of statements, where statements are labeled in numerical order. Return sets of labels where each set contains similar statements.

<Statement List>
{statement_list}
</Statement List>""",

    "combine_ac_system_prompt": """You are a helpful assistant.""",
    "combine_ac_user_prompt": """You are given a list of statements, where each statement is similar to each other but not necessarily identical. Remove redundant information and return 1-3 concise and clear statements.

<Statement List>
{statement_list}
</Statement List>""",
}

responder_prompts = {
    "respond": """Answer the following question based on the given context. Format your answer in one sentence:

Context: {context}

Question: {question}

Answer: """,

    "impact": """You will be given a statement and a context. Suppose the statement is TRUE, how much of the context will you change to keep it consistent with the statement?
Your final answer should be a percentage number between 0 and 100, representing the percentage of the context you will change. Return ONLY the percentage number, nothing else.

<Statement>
{statement}
</Statement>

<Context>
{context}
</Context>""",

    "impact2": """You will be given a statement and a context. Please estimate how much of the context contradicts the statement?
Your final answer should be a percentage number between 0 and 100, representing the percentage of the context that contradicts the statement. Return ONLY the percentage number, nothing else.

<Statement>
{statement}
</Statement>

<Context>
{context}
</Context>"""

}

uncertainty_metrics_prompts = {
    "self_consistency_system_prompt": """You are a helpful assistant who carefully examines difference between statements.""",
    "self_consistency_user_prompt": """Given a test claim and a list of candidate claims, rate the level of consistency between the test claim and candidate claims. Pay special attention to factual consistency. Return a score between 0 and 10, representing the level of consistency. Return ONLY the score, nothing else.

<Test Claim>
{test_claim}
</Test Claim>

<Candidate Claims>
{candidate_claims}
</Candidate Claims>""",
}

evaluator_prompts = {
    "eval_claims_from_reference_system_prompt": """You are a meticulous fact-checker who checks the correctness of claims based on reference documents.""",
    "eval_claims_from_reference_user_prompt": """Based solely on the reference passage, evaluate the correctness of each claim. For each claim, return only one of the following: "correct", "incorrect", or "not_enough_information", along with the index of the claim.

<Topic>{topic}</Topic>
<Reference>{reference}</Reference>
<Claims>
{claims}
</Claims>""",

    "Deprecated_eval_claims_from_reference_user_prompt_single": """Based solely on the Reference passage, evaluate whether the Claim is relevant to the passage. If the Claim is relevant, is it correct according to the passage? Choose your answer from <correct/incorrect/irrelevant>.

<Topic>{topic}</Topic>
<Claim>{claim}</Claim>
<Reference>{reference}</Reference>""",

    "eval_claims_from_reference_user_prompt_single": """Is the following claim supported by the reference passage? Choose your answer from <supported/not supported>.

<Claim>{claim}</Claim>

<Reference>{reference}</Reference>""",

    "from_generations_system_prompt": """You are a meticulous fact-checker who checks the correctness of claims based on a given passage.""",
    "from_generations_user_prompt": """Is the following claim supported by the given passage?

<Claim>
{claim}
</Claim>

<Passage>
{passage}
</Passage>""",
}

self_eval_prompts = {
    "extract_atomic_facts": """Given the following contexual question, context, and a specific sentence from the context, extract simple and standalone atomic facts from that sentence.
Each fact should be a standalone statement.
Extract no more than 3 atomic facts from the given sentence.
Output only the facts as a bullet point list.

<Example>
<Contexual Question>What was Barack Obama's career path before becoming president?</Contexual Question>
<Context>Barack Obama served two terms as president from 2009 to 2017. Before his presidency, he graduated from Harvard Law School in 1991 and worked as a civil rights attorney in Chicago, where he also taught constitutional law at the University of Chicago Law School from 1992 to 2004.</Context>
<Sentence to Analyze>Before his presidency, he graduated from Harvard Law School in 1991 and worked as a civil rights attorney in Chicago, where he also taught constitutional law at the University of Chicago Law School from 1992 to 2004.</Sentence to Analyze>
Facts from the given sentence:
• He graduated from Harvard Law School in 1991
• He worked as a civil rights attorney in Chicago
• He taught constitutional law at the University of Chicago Law School from 1992 to 2004
</Example>

<Contexual Question>{question}</Contexual Question>
<Context>{context}</Context>
<Sentence to Analyze>{sentence}</Sentence to Analyze>
Facts from the given sentence:""",


    "decompose_generation": """Given context and a paragraph of text, deconstruct the text into the smallest possible standalone and self-contained facts without semantic repetition.
Each fact should come from the text and must be related to the context. An example is provided below:

<Example>
<Context>Give me a bio of Barack Obama</Context>
<Text>Barack Obama is an American politician and attorney who served as the 44th president of the United States from 2009 to 2017. Born on August 4, 1961, in Honolulu, Hawaii, he graduated from Columbia University and Harvard Law School. 
In 2009 he was awarded the Nobel Peace Prize "for his extraordinary efforts to strengthen international diplomacy and cooperation between peoples."</Text>
<Facts>
1. Barack Obama is an American politician and attorney.
2. Barack Obama served as the 44th president of the United States from 2009 to 2017.
3. Barack Obama was born on August 4, 1961, in Honolulu, Hawaii.
4. Barack Obama graduated from Columbia University and Harvard Law School.
5. Barack Obama was awarded the Nobel Peace Prize in 2009.
</Facts>
</Example>

Now, deconstruct the following text into facts, format your response as a numbered list of facts only, with no additional text:

<Context>{context}</Context>
<Text>{text}</Text>
<Facts>""",


    "extract_ac_system_prompt": """Extract factual information from given text.""",
    "extract_ac_user_prompt": """Given context and a paragraph of text, deconstruct the text into the smallest possible standalone and self-contained facts without semantic repetition. Each fact should come from the text and must be related to the context.

<Context>{context}</Context>
<Text>{text}</Text>""",


#     "q_from_claims_system_prompt": """You are a helpful documentator who breaks down statements into smaller units in the form of questions.""",
#     "q_from_claims_user_prompt": """Given context and a list of claims, generate specific, clear questions that have their answers contained in the corresponding claims.
# For each claim, generate 1-3 questions that ask for factual information. The generated questions should start from asking the most general information and must be related to the context.

# <Example>
# <Context>Give me a bio of Roger Federer</Context>
# <Claims>
# claim 1: "Roger Federer is a retired Swiss professional tennis player, widely regarded as one of the greatest in the sport's history."
# claim 2: "Born on August 8, 1981, in Basel, Switzerland, he turned professional in 1998 and went on to win 20 Grand Slam singles titles, including a record eight Wimbledon titles."
# claim 3: "Beyond tennis, he is involved in philanthropy through the Roger Federer Foundation, which supports educational initiatives in Africa and Switzerland."
# </Claims>

# <Questions>
# from_claim 1: What nationality is Roger Federer?
# from_claim 1: What sport did Roger Federer play?
# from_claim 1: Is Roger Federer retired?
# from_claim 2: When and where was Roger Federer born?
# from_claim 2: When did Roger Federer turn professional?
# from_claim 2: How many Grand Slam titles has Roger Federer won?
# from_claim 3: What area does Roger Federer support through philanthropy?
# </Questions>
# </Example>

# Now, generate questions for the following claims:

# <Context>{context}</Context>
# <Claims>
# {claims}
# </Claims>""",

    "generate_atomic_questions": """Generate specific, focused questions that can be answered by each of the following facts.
Each question should be clear and direct. Use the contexual question for reference if needed.

<Example>
Contexual Question: "Give me a bio of Barack Obama"
Facts:
Fact 1: "Barack Obama was the 44th president of the United States."
Fact 2: "He was born in Hawaii."
Fact 3: "He went to Harvard Law School."
Fact 4: "Obama was awarded the 2009 Nobel Peace Prize for efforts in international diplomacy."

Questions:
Question 1: "Who was the 44th president of the United States?"
Question 2: "Where was Barack Obama born?"
Question 3: "Which college did Barack Obama go to?"
Question 4: "Did Barack Obama win the Nobel Peace Prize?"
</Example>

Now, generate one question for each of the following facts:
Contexual Question: "{context_q}"
Facts:
{facts_list}

Format your response as a numbered list of questions only, with no additional text:
Question 1. [Question for Fact 1]
Question 2. [Question for Fact 2]
etc.
""",


    "generate_atomic_questions_no_facts_extraction": """Generate specific, focused "atomic question" that can be answered by the fact provided.
MAKE SURE the atomic question is related to the context. The atomic question should be clear and direct. You can generate more than one atomic question.

<Example>
Context: "Give me a bio of Barack Obama"
Fact: "Barack Obama was the 44th and president of the United States began with his first inauguration on January 20, 2009."

Atomic Questions:
1. Who was the 44th president of the United States?
2. When did the 44th president of the United States take office?
</Example>

Now, generate atomic question(s) for the following fact:
Context: "{context}"
Fact: "{fact}"

Format your response as a numbered list of questions only, with no additional text:
Question 1. [Atomic Question 1]
Question 2. [Atomic Question 2]
etc.
""",


#     "deduplicate_questions": """Given the following list of questions, identify groups of semantically equivalent questions.
# Two questions are semantically equivalent if they ask for the same information, even if phrased differently.
# For each group of equivalent questions, select the clearest and most concise question to represent that group and remove the duplicates.

"deduplicate_questions": """Given the following list of questions, remove duplicates. Two questions are duplicates if they are semantically equivalent, asking for the same information.
Return a numbered list of filtered questions.

<Example>
<Questions>
1. Who was the 44th president of the United States?
2. Where was Barack Obama born?
3. In what location was Barack Obama's birthplace?
4. Who served as the 44th US president?
5. What law school did Barack Obama attend?
</Questions>

<Filtered Questions>
1. Who was the 44th president of the United States?
2. Where was Barack Obama born?
3. What law school did Barack Obama attend?
</Filtered Questions>
</Example>

<Questions>
{questions}
</Questions>

<Filtered Questions>""",


    "generate_one_answer_to_atomic_questions": """Given a topic and a question, answer the question in at most one sentence.

<Example>
Topic: "Give me a bio of Barack Obama"
Question: "What did Barack Obama do before he became president?"
Answer: "Barack Obama was a civil rights attorney in Chicago."
</Example>

Now, answer the following question:

Topic: "{topic}"
Question: "{question}"

Format your response as a single answer only, with no additional text:

Answer: """,

    "generate_multiple_answers_to_atomic_questions": """Given a topic and a list of specific questions, generate answers to each of the questions. Limit each of your answers to at most one sentence.

<Example>
Topic: "Give me a bio of Barack Obama"
Questions:
1. What did Barack Obama do before he became president?
2. Where was Barack Obama born?
3. Which college did Barack Obama go to?
Answers:
1. Barack Obama was a civil rights attorney in Chicago.
2. Barack Obama was born in Hawaii.
3. Barack Obama went to Harvard Law School.
</Example>

Now, generate answers to the following questions:

Topic: "{topic}"
Questions:
{questions}

Format your response as a numbered list of answers only, with no additional text:

Answers:
""",
}