import answerer

if __name__ == "__main__":
    answerer = answerer.Answerer()
    print("Building Model!")
    answerer.build_model()
    print("Initializing Done!")
    print(answerer.answer("covid urine").head(5))
