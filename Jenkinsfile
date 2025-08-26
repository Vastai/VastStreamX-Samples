pipeline {
  agent {
    kubernetes {
      yaml """
      apiVersion: v1
      kind: Pod
      spec:
        containers:
          - name: container-lowt4f
            imagePullPolicy: IfNotPresent
            tty: true
            image: 'harbor.vastaitech.com/software/vsxs:last'
            resources:
              requests:
                cpu: '2'
                memory: 1000Mi
              limits:
                cpu: '2'
                memory: 1000Mi
            volumeMounts:
              - name: host-time
                mountPath: /etc/localtime
                readOnly: true
              - name: qa-data
                readOnly: false
                mountPath: /qa-data/
        initContainers: []
        volumes:
          - hostPath:
              path: /etc/localtime
              type: ''
            name: host-time
          - hostPath:
              path: /data/qa-data_ce
            name: qa-data
        nodeSelector:
          kubernetes.io/hostname: node20-103
      metadata: {}
      """
      inheritFrom 'jnlp'
      defaultContainer 'container-lowt4f'
      namespace 'internal-release-pipelinesb4t9l'
    }

  }
  stages {
    stage('stage-build') {
      agent {
        kubernetes {
          inheritFrom 'jnlp'
          namespace 'internal-release-pipelinesb4t9l'
          yaml """
          apiVersion: v1
          kind: Pod
          spec:
            volumes:
              - hostPath:
                  path: /data/qa-data_ce
                name: qa-test
              - hostPath:
                  path: /etc/localtime
                  type: ''
                name: host-time
            containers:
              - name: container-vsxs-test
                imagePullPolicy: IfNotPresent
                tty: true
                image: 'harbor.vastaitech.com/software/vsxs:last'
                resources:
                  requests:
                    cpu: '48'
                    memory: 128000Mi
                  limits:
                    cpu: '48'
                    memory: 128000Mi
                volumeMounts:
                  - name: host-time
                    mountPath: /etc/localtime
                    readOnly: true
                  - name: qa-test
                    readOnly: false
                    mountPath: /qa-data/
            initContainers: []
            nodeSelector:
              kubernetes.io/hostname: node28-86
          metadata: {}
          """
          defaultContainer 'container-vsxs-test'
        }

      }
      steps {
        withCredentials([sshUserPrivateKey(credentialsId : 'gqzhou-ssh' ,keyFileVariable : 'SSH_KEY' ,passphraseVariable : 'SSH_PASSWORD' ,usernameVariable : 'SSH_USERNAME' ,)]) {
          sh '''
        eval `ssh-agent`
        ssh-add $SSH_KEY
        ssh-add -l
        ssh -o "StrictHostKeyChecking no" git@192.168.20.70
        echo $SSH_KEY >> ssh_key.txt

        pwd
        echo ">>> build env"
        echo $RUN_NODE_NAME
        wget -q http://release.vastai.com/infra/libopenblas.a
        cp -f libopenblas.a /opt/openblas/lib/libopenblas.a
        uname -a
        gcc --version
        cmake --version
        git clone git@192.168.20.70:AIS/cicd.git
        rm -rf build && mkdir build
        cd build
        cmake ../cicd/ -DBUILD_PROJECT=vsxs -DVSXSBRANCH=${GIT_BRANCH}
        make install -j 64
        make package
        mkdir -p /qa-data/cicd/${BUILD_TAG}/
        cp vsxs-*.tar.gz /qa-data/cicd/${BUILD_TAG}/
        '''
        }

      }
    }

    stage('test') {
      parallel {
        stage('stage-test-module') {
          agent {
            kubernetes {
              inheritFrom 'jnlp'
              namespace 'internal-release-pipelinesb4t9l'
              yaml '''
          apiVersion: v1
          kind: Pod
          spec:
            volumes:
              - hostPath:
                  path: /data/qa-data_ce
                name: qa-test
              - hostPath:
                  path: /etc/localtime
                  type: \'\'
                name: host-time
            containers:
              - name: container-vsxs-test
                imagePullPolicy: IfNotPresent
                tty: true
                image: \'harbor.vastaitech.com/software/vsxs:last\'
                privileged: true
                resources:
                  requests:
                    cpu: \'100\'
                    memory: 500000Mi
                  limits:
                    cpu: \'100\'
                    memory: 500000Mi
                volumeMounts:
                  - name: host-time
                    mountPath: /etc/localtime
                    readOnly: true
                  - name: qa-test
                    readOnly: false
                    mountPath: /qa-data/
                env:
                  - name: VASTAI_VISIBLE_DEVICES
                    value: all
            initContainers: []
            nodeSelector:
              kubernetes.io/hostname: node28-86
          metadata: {}
          '''
              defaultContainer 'container-vsxs-test'
            }

          }
          steps {
            script {
              catchError(buildResult: 'SUCCESS', catchInterruptions: false) {
                sh '''
                wget -O 'ai-v2.8.1-20250226-5-linux-x86_64-sdk2-python3.8.bin'  http://devops.vastai.com/kapis/artifact.kubesphere.io/v1alpha1/artifact?artifactid=4976
                chmod a+x ai-v2.8.1-20250226-5-linux-x86_64-sdk2-python3.8.bin
                ./ai-v2.8.1-20250226-5-linux-x86_64-sdk2-python3.8.bin
cp /qa-data/cicd/${BUILD_TAG}/vsxs-*.tar.gz .
if [ -d pro_test ];then rm -rf pro_test;fi
mkdir -p pro_test
tar -zxvf vsxs-*.tar.gz -C pro_test
cd pro_test
cd *
./run_main.py config_test -i "AI_INTEGRATION" -s ${BUILD_TAG} -b vsxs -l vsxs_ai -p vsxs -a vsxs -m Continuous run_test -s 44 service_test -s "guoqiang.zhou,zhonghong.tan"
'''
              }
            }

          }
        }

        stage('stage-test-media') {
          agent {
            kubernetes {
              inheritFrom 'jnlp'
              namespace 'internal-release-pipelinesb4t9l'
              yaml '''apiVersion: v1
kind: Pod
spec:
  volumes:
    - hostPath:
        path: /data/qa-data_ce
      name: qa-test
    - hostPath:
        path: /etc/localtime
        type: \'\'
      name: host-time
  containers:
    - name: container-nak2eb
      imagePullPolicy: IfNotPresent
      tty: true
      image: \'harbor.vastaitech.com/software/vsxs:last\'
      resources:
        requests:
          cpu: \'48\'
          memory: 128000Mi
        limits:
          cpu: \'48\'
          memory: 128000Mi
      env:
        - name: VASTAI_VISIBLE_DEVICES
          value: all
      volumeMounts:
        - name: host-time
          mountPath: /etc/localtime
          readOnly: true
        - name: qa-test
          readOnly: false
          mountPath: /qa-data/
  initContainers: []
  nodeSelector:
    kubernetes.io/hostname: node20-103
metadata: {}
'''
              defaultContainer 'container-nak2eb'
            }

          }
          steps {
            script {
              catchError(buildResult: 'SUCCESS', catchInterruptions: false) {
                sh '''
                wget -O 'ai-v2.8.1-20250226-5-linux-x86_64-sdk2-python3.8.bin'  http://devops.vastai.com/kapis/artifact.kubesphere.io/v1alpha1/artifact?artifactid=4976
                chmod a+x ai-v2.8.1-20250226-5-linux-x86_64-sdk2-python3.8.bin
                ./ai-v2.8.1-20250226-5-linux-x86_64-sdk2-python3.8.bin
cp /qa-data/cicd/${BUILD_TAG}/vsxs-*.tar.gz .
if [ -d pro_test ];then rm -rf pro_test;fi
mkdir -p pro_test
tar -zxvf vsxs-*.tar.gz -C pro_test
cd pro_test
cd *
./run_main.py config_test -i "CODEC_INTEGRATION" -s ${BUILD_TAG} -b vsxs -l vsxs_meida -p vsxs -a vsxs -m Experimental run_test -s 44 service_test -s "guoqiang.zhou"
'''
              }
            }

          }
        }

      }
    }

    stage('stage-release-package') {
      agent none
      steps {
        sh '''mkdir -p ${WORKSPACE}/package_release
find /qa-data/cicd/${BUILD_TAG}/ -type f|grep -v "sh$"|xargs -i cp {} ${WORKSPACE}/package_release'''
        vastaiBinaryArtifact(path: '${WORKSPACE}/package_release')
      }
    }

  }
}
