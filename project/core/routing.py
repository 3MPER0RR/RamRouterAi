class Router:

    @staticmethod
    def decide(score):

        if score < 0.25:
            return "local"

        elif score < 0.50:
            return "memory"

        elif score < 0.75:
            return "tool"

        return "llm"
