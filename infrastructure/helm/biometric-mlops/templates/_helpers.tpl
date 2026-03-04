{{/*
[Phase 6c] Helm chart helpers
*/}}
{{- define "biometric-mlops.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "biometric-mlops.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "biometric-mlops.namespace" -}}
{{- default "biometric-mlops" .Values.namespaceOverride }}
{{- end }}

{{- define "biometric-mlops.labels" -}}
app.kubernetes.io/name: {{ include "biometric-mlops.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: bosch-mlops-evaluation
app: biometric-mlops
{{- end }}
