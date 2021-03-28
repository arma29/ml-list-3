import src.processing as processing
import src.training as training
import src.reporting as reporting


# Iterações = 100, alpha = 0.1, window_width = 0.2, epslon = 0.1

def main():
    raw_lst = processing.get_raw_data()
    names_lst = processing.get_raw_names()

    # for idx in range(len(raw_lst)):
    for idx in range(1):
        dataset_name = names_lst[idx]

        X, y, target_names = processing.process_data(raw_lst[idx])

        model_dict = training.train_model(X, y, target_names, dataset_name)

        training.print_elapsed_time(model_dict)

        reporting.produce_report(model_dict)


if __name__ == "__main__":
    main()
