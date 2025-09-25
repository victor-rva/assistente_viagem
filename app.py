import os
from dotenv import load_dotenv
load_dotenv()

# ChatGoogleGenerativeAI: Classe para interagir com os modelos de chat da Google/Gemini.
from langchain_google_genai import ChatGoogleGenerativeAI
# ChatPromptTemplate: Classe para criar templates de prompt para modelos de chat.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# RunnableWithMessageHistory: Adiciona gerenciamento de histórico de mensagens a uma chain.
from langchain_core.runnables.history import RunnableWithMessageHistory
# BaseChatMessageHistory: Classe base para histórico de mensagens.
from langchain_core.chat_history import BaseChatMessageHistory
# ChatMessageHistory: Implementação em memória para o histórico de mensagens.
from langchain_community.chat_message_histories import ChatMessageHistory

template = """Você é um assistente de viagem que ajuda os usuários a planejar suas viagens, dando sugestões de destinos, roteiros e dicas práticas. Seja amigável e prestativo.
A primeira coisa que deve fazer é perguntar para onde o usuário quer viajar, com quantas pessoas e por quanto tempo."""

# Estrutura do prompt que será enviado ao modelo
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    # O MessagesPlaceholder é o local onde o histórico da conversa (gerenciado pelo Runnable) será inserido.
    MessagesPlaceholder(variable_name="history"),
    # A entrada atual do usuário.
    ("human", "{input}"),
])

# O modelo (llm) deve ser uma instância de uma classe de modelo, como ChatOpenAI.
# A temperatura controla a aleatoriedade das respostas (0.7 é um bom equilíbrio).
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
# Cria a cadeia de processamento (chain) usando LangChain Expression Language (LCEL)
# O fluxo é: o input do usuário preenche o `prompt`, que é então enviado para o `llm`.
chain = prompt | llm

# Dicionário em memória para armazenar o histórico de conversas por sessão.
# Em uma aplicação real, isso poderia ser substituído por um banco de dados (Redis, SQL, etc.).
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Busca ou cria um histórico de chat para uma determinada ID de sessão.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Cria o gerenciador de histórico, envolvendo a chain principal.
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    # Chave para a entrada do usuário no dicionário de input.
    input_messages_key="input",
    # Chave onde o histórico de mensagens será armazenado e lido. O nome deve corresponder
    # ao `variable_name` do MessagesPlaceholder no prompt.
    history_messages_key="history"
)

def iniciar_assistente_viagem():
    """
    Função principal para iniciar a interação com o assistente no terminal.
    """
    print("Bem-vindo ao Assistente de Viagem! Digite 'sair' para encerrar a conversa.\n")
    session_id = "user123" # ID de sessão fixa para este exemplo

    while True:
        pergunta_usuario = input("Você: ")
        if pergunta_usuario.lower() in ['sair', 'exit', 'quit']:
            print("Assistente: Obrigado por usar o Assistente de Viagem. Boa viagem!")
            break

        # Invoca a chain com histórico.
        # O input é um dicionário contendo a chave `input_messages_key` definida acima.
        # A configuração 'configurable' é usada pelo RunnableWithMessageHistory para identificar a sessão.
        resposta = chain_with_history.invoke(
            {"input": pergunta_usuario},
            config={"configurable": {"session_id": session_id}}
        )

        # O resultado de uma chain com um modelo de chat é um objeto de mensagem (AIMessage).
        # Acessamos o conteúdo de texto com o atributo `.content`.
        print(f"Assistente de Viagem: {resposta.content}")

if __name__ == "__main__":
    iniciar_assistente_viagem()