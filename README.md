{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPm5yBx2busSxyJ+BGc0+7f",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/pulipatikarthikeya07/ML-Foundations-Projects/blob/main/README.md\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UpMNMfwYemL6",
        "outputId": "f0105a96-6c3d-476d-98c4-11bb7911f1ce"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Data preprocessing complete. Training shape: (8, 3)\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import mean_absolute_error\n",
        "\n",
        "# 1. Create a dummy real estate dataset\n",
        "data = {\n",
        "    'sqft': [1500, 1800, 2400, 3000, 1200, 1700, 2100, 2800, 1350, 2200],\n",
        "    'bedrooms': [3, 3, 4, 4, 2, 3, 3, 4, 2, 4],\n",
        "    'age_years': [5, 10, 1, 12, 15, 8, 3, 6, 20, 2],\n",
        "    'price_usd': [250000, 290000, 390000, 450000, 180000, 275000, 340000, 430000, 195000, 365000]\n",
        "}\n",
        "\n",
        "df = pd.DataFrame(data);\n",
        "\n",
        "# 2. Separate features (X) from the target variable (y)\n",
        "X = df[['sqft', 'bedrooms', 'age_years']]\n",
        "y = df['price_usd']\n",
        "\n",
        "# 3. Split data into Training set (80%) and Testing set (20%)\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# 4. Feature Scaling (Normalizes the range of independent variables)\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)\n",
        "\n",
        "print(\"Data preprocessing complete. Training shape:\", X_train_scaled.shape)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 5. Initialize and train the Linear Regression Model\n",
        "model = LinearRegression()\n",
        "model.fit(X_train_scaled, y_train)\n",
        "\n",
        "# 6. Predict on the test data\n",
        "predictions = model.predict(X_test_scaled)\n",
        "\n",
        "# 7. Evaluate using Mean Absolute Error (MAE)\n",
        "mae = mean_absolute_error(y_test, predictions)\n",
        "\n",
        "print(f\"Model Training Successful!\")\n",
        "print(f\"Actual Prices: {list(y_test)}\")\n",
        "print(f\"Predicted Prices: {[round(p, 2) for p in predictions]}\")\n",
        "print(f\"Mean Absolute Error: ${round(mae, 2)}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IHXKRs_nf1B2",
        "outputId": "616d4b6a-5b48-4d5a-d3bb-5eb600b2ed2d"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model Training Successful!\n",
            "Actual Prices: [195000, 290000]\n",
            "Predicted Prices: [np.float64(190095.3), np.float64(281754.63)]\n",
            "Mean Absolute Error: $6575.03\n"
          ]
        }
      ]
    }
  ]
}