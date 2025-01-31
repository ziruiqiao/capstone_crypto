from llm_tools import *


def main():
    while True:
        user_input = input("Please Enter the Cryptos to be analyzed: ")
        if user_input != "q" or user_input != "quit":
            output = portfolio_suggest_agent(user_input)
            print(output)
        else:
            return


if __name__ == "__main__":
    main()
