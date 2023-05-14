import matplotlib.pyplot as plt
import logging
import argparse
import pandas as pd
import os
import seaborn as sns

# Configure logging
logging.basicConfig(
    filename="plot.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def plot_histograms(df, plot_dir):
    categories = [
        ("EAR Quality", ["Interesting", "AbtEarStudy"]),
        ("How", ["Alone", "WithOne", "WithGroup", "Talk", "Phone"]),
        (
            "With Whom",
            [
                "Self",
                "ExPartner",
                "NewPartner",
                "Child",
                "OtherFamily",
                "FriAcq",
                "Stranger",
                "Pet",
            ],
        ),
        (
            "Conversation Type",
            ["SmallTalk", "SubConvo", "PersEmoDis", "Gossip", "Practical"],
        ),
        (
            "Interaction Quality",
            [
                "Gratitude",
                "Spirituality",
                "Blame",
                "ComplainWhine",
                "Affection",
                "PosSupRec",
                "NegSupRec",
            ],
        ),
        (
            "Activity",
            [
                "SocEnt",
                "Intoxicated",
                "Working",
                "Housework",
                "Church",
                "Hygiene",
                "EatDrink",
                "Sleep",
            ],
        ),
        ("Emotion", ["Laugh", "Cry", "Sing", "MadArgue", "Sigh", "Yawn"]),
    ]

    for category, subcategories in categories:
        fig, ax = plt.subplots()

        for subcategory in subcategories:
            # Skip if the subcategory is not in the dataframe
            if subcategory not in df.columns:
                continue

            # Filter data for the specific subcategory
            subcategory_df = df[df[subcategory] == 1]

            # Create a histogram for the yamnet_tv score of the subcategory
            ax.hist(
                subcategory_df["yamnet_tv"],
                bins=50,
                alpha=0.5,
                label=f"{subcategory} = 1",
            )

        plt.xlabel("yamnet_tv")
        plt.ylabel("Count")
        plt.title(f"Yamnet TV detector performance histogram ({category})")
        plt.legend(loc="upper right")

        # Save the plot to a file in 'plot_dir'
        plt.savefig(f"{plot_dir}/histogram_{category}.png", bbox_inches="tight")
        plt.close(fig)


def plot_results(df, plot_dir):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    sns.kdeplot(
        df[df["Tv"] == 0]["yamnet_tv"],
        shade=True,
        alpha=0.5,
        label="Tv = 0",
        color="green",
        ax=axs[0],
    )
    axs[0].set_xlabel("Yamnet TV Probability")
    axs[0].set_ylabel("Density")
    axs[0].set_title("Yamnet TV detector performance density plot (Tv = 0)")
    axs[0].legend(loc="upper right")

    sns.kdeplot(
        df[df["Tv"] == 1]["yamnet_tv"],
        shade=True,
        alpha=0.5,
        label="Tv = 1",
        color="red",
        ax=axs[1],
    )
    axs[1].set_xlabel("Yamnet TV Probability")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Yamnet TV detector performance density plot (Tv = 1)")
    axs[1].legend(loc="upper right")

    plt.tight_layout()

    # Create 'plot_dir' if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot to a file in 'plot_dir'
    plt.savefig(f"{plot_dir}/density_plot.png")


def initialize_args(parser):
    # Input paths
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the directory containing data",
    )
    parser.add_argument(
        "--plot_dir", required=True, help="Path to the directory to save plots to"
    )


def main(args):
    logging.info("Plotting...")

    data_dir = args.data_dir
    plot_dir = args.plot_dir

    df = pd.read_csv(data_dir)
    plot_results(df, plot_dir)
    logging.info("Finished processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
