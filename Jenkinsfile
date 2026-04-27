pipeline {
    agent any

    stages {

        stage('Clone Repo') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Sanity Check') {
            steps {
                sh '''
                . venv/bin/activate
                python -c "import fastapi, sklearn, mlflow, prometheus_client"
                '''
            }
        }

        stage('Docker Test') {
            steps {
                sh 'docker ps'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t mlops-confidence-project -f docker/Dockerfile .'
            }
        }

    }
}
