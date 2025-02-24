from langchain.prompts import PromptTemplate

rhetorical_role_identify_prompt = PromptTemplate(
    input_variables=["user_question"],
    template="""
        Given the user query below, determine which rhetorical roles from this set:
        [FAC, ARG, PRE, Ratio, RLC, STA]
        are most relevant to answering it. 
        
        These roles are given as:
        FAC (Facts of the Case): A summary of the key events and circumstances leading to the legal dispute.
        RLC (Ruling by Lower Court): The decision made by the lower court before the case reached the present court.
        RPC (Ruling by Present Court): The final decision or judgment issued by the current court handling the case.
        ARG (Argument): The reasoning and legal contentions presented by the parties involved in the case.
        Ratio (Ratio Decidendi): The legal principle or reasoning that forms the basis for the court’s decision.
        PRE (Precedent): Previous judicial decisions cited to support legal arguments or rulings.
        STA (Statute): A formal written law enacted by a legislative authority relevant to the case.
    
        
        Return them as a valid Python list, e.g. ["FAC", "ARG"] or [] if none are relevant.
        
        User Query: {user_question}
        ---
        Answer:
    """
)


generation_prompt_rag = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        You are a helpful legal AI assistant. 
        Below is some contextual text from a legal document, grouped by rhetorical role.
        
        Context:
        {context}
        
        The user asked:
        "{question}"
        
        Based on the provided context, give a concise answer:
        """
)

def build_prompt_for_generation(question: str, retrieved_lines: list):
    """
    Here we will build a prompt from the given question.
    We should ideally first ask what rhetorical role is the question about.
    We can then pass that role along with the
    Build a final prompt that includes the question + relevant lines.
    You can group them by rhetorical role if you like.
    """
    roles_map = {}
    for line_info in retrieved_lines:
        role = line_info["metadata"]["rhetorical_role"]
        text = line_info["text"]
        roles_map.setdefault(role, []).append(text)

    # Now build a prompt that organizes lines by role
    context_str = ""
    for role, texts in roles_map.items():
        context_str += f"\n--- {role} ---\n"
        for t in texts:
            context_str += f"- {t}\n"

    prompt = f"""
        You are a helpful legal AI assistant working with Supreme Court case documents. 
        The user asked: "{question}"

        Here are some relevant lines from the case documents. Also included is the rhetorical role:

        {context_str}

        Please provide a concise answer based on the relevant lines. 
        If you need any direct quotes, reference them as needed.
        """
    return prompt


def build_prompt_for_role_identification(question: str):
    """
    We will ask the model to give us a rhetorical role from the question first.
   This role is then passed for fetching the top documents.
    """
    prompt = f"""
        You are a helpful legal AI assistant working with Supreme Court case documents. 
        The user asked: "{question}"

        Given the following 7 rhetorical roles: 

            FAC (Facts of the Case): A summary of the key events and circumstances leading to the legal dispute.
            RLC (Ruling by Lower Court): The decision made by the lower court before the case reached the present court.
            RPC (Ruling by Present Court): The final decision or judgment issued by the current court handling the case.
            ARG (Argument): The reasoning and legal contentions presented by the parties involved in the case.
            Ratio (Ratio Decidendi): The legal principle or reasoning that forms the basis for the court’s decision.
            PRE (Precedent): Previous judicial decisions cited to support legal arguments or rulings.
            STA (Statute): A formal written law enacted by a legislative authority relevant to the case.

        Please tell which rhetorical role is this question related to. 
        There can be multiple roles, so please give your answer as a comma separated list.
        If the question is only relevant to one role, give your answer as a list with single element. 
        """
    return prompt
