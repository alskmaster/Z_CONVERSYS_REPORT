# -*- coding: utf-8 -*-
# ==============================================================================
# ZABBIX REPORTER - ENTERPRISE EDITION v20.0 FINAL
#
# Autor: Marcio Bernardo, Conversys IT Solutions
# Data: 14/08/2025
# Descrição: Versão final com controle de paginação via CSS. Garante que
#            seções principais comecem em novas páginas e que blocos de
#            conteúdo (como KPIs) não sejam quebrados, resultando em um
#            documento profissional e bem formatado.
# ==============================================================================

# --- Importações Essenciais ---
import os
import sys
import json
import base64
import uuid
import logging
import re
import textwrap
import datetime as dt
import time
from io import BytesIO
from collections import defaultdict
from functools import wraps
import threading
import traceback

# --- Dependências (instale com: pip install -r requirements.txt) ---
# Flask, Flask-SQLAlchemy, Flask-Login, python-dotenv, Werkzeug, Jinja2,
# requests, pandas, matplotlib, xhtml2pdf, urllib3, PyPDF2

# --- Importações de Bibliotecas ---
from flask import (Flask, render_template_string, request, flash, redirect,
                   url_for, send_file, send_from_directory, get_flashed_messages, g, jsonify, session)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, login_user, logout_user, login_required,
                       current_user, UserMixin)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import requests
import pandas as pd
from jinja2 import Template, DictLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xhtml2pdf import pisa
from PyPDF2 import PdfWriter, PdfReader, errors as PyPDF2Errors

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# --- Carregamento de Configurações ---
load_dotenv()

class Config:
    """Configurações da aplicação Flask e Zabbix."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'mude-esta-chave-secreta-em-producao-agora'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///zabbix_reporter_v20.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'uploads'
    GENERATED_REPORTS_FOLDER = 'relatorios_gerados'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'html', 'pdf'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ZABBIX_URL = os.getenv('ZABBIX_URL')
    ZABBIX_USER = os.getenv('ZABBIX_USER')
    ZABBIX_PASSWORD = os.getenv('ZABBIX_PASSWORD')
    ZABBIX_TOKEN = os.getenv('ZABBIX_TOKEN')
    SUPERADMIN_PASSWORD = os.getenv('SUPERADMIN_PASSWORD') or 'admin123'

# --- Inicialização da Aplicação e Extensões ---
app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = "Por favor, faça login para acessar esta página."
login_manager.login_message_category = "info"

# --- Modelos do Banco de Dados (SQLAlchemy) ---

user_client_association = db.Table('user_client_association',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('client_id', db.Integer, db.ForeignKey('client.id'), primary_key=True)
)

class SystemConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(100), default="Conversys IT Solutions")
    footer_text = db.Column(db.String(255), default=f"Conversys IT Solutions Copyright {dt.datetime.now().year} – Todos os direitos Reservados | Política de privacidade")
    primary_color = db.Column(db.String(7), default="#2c3e50")
    secondary_color = db.Column(db.String(7), default="#3498db")
    logo_dark_bg_path = db.Column(db.String(255), nullable=True)
    logo_light_bg_path = db.Column(db.String(255), nullable=True)
    logo_size = db.Column(db.Integer, default=50)
    login_media_path = db.Column(db.String(255), nullable=True)
    login_media_fill_mode = db.Column(db.String(10), default='cover')
    login_media_bg_color = db.Column(db.String(7), default='#2c3e50')
    report_cover_path = db.Column(db.String(255), nullable=True)
    report_final_page_path = db.Column(db.String(255), nullable=True)

class Role(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    zabbix_group_id = db.Column(db.String(50), nullable=True)
    sla_contract = db.Column(db.Float, nullable=False, default=99.9)
    logo_path = db.Column(db.String(255), nullable=True)
    reports = db.relationship('Report', backref='client', lazy=True, cascade="all, delete-orphan")

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    role_id = db.Column(db.Integer, db.ForeignKey('role.id'), nullable=False)
    role = db.relationship('Role', backref='users')
    reports = db.relationship('Report', backref='author', lazy=True)
    clients = db.relationship('Client', secondary=user_client_association, lazy='subquery',
                              backref=db.backref('users', lazy=True))

    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)
    def has_role(self, role_name): return self.role.name == role_name

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), unique=True, nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)
    reference_month = db.Column(db.String(7), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    username = db.Column(db.String(64), nullable=False)
    action = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=dt.datetime.utcnow)
    user = db.relationship('User', backref='audit_logs')

# --- Camada de Serviço e Lógica de Negócio ---

REPORT_GENERATION_TASKS = {}
TASK_LOCK = threading.Lock()

class AuditService:
    @staticmethod
    def log(action, user=None):
        log_user = user or (current_user if current_user.is_authenticated else None)
        username = log_user.username if log_user else "Anonymous"
        user_id = log_user.id if log_user else None
        new_log = AuditLog(user_id=user_id, username=username, action=action)
        db.session.add(new_log)
        db.session.commit()

# --- LÓGICA DE CONEXÃO E COLETA ZABBIX ---

def update_status(task_id, message):
    with TASK_LOCK:
        if task_id in REPORT_GENERATION_TASKS:
            REPORT_GENERATION_TASKS[task_id]['status'] = message
            app.logger.info(f"TASK {task_id}: {message}")

def fazer_request_zabbix(body, url, allow_retry=True):
    headers = {'Content-Type': 'application/json-rpc', 'Accept-Encoding': 'gzip'}
    max_retries = 2 if allow_retry else 1
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(body), verify=False, timeout=120)
            if response.status_code >= 500 and attempt < max_retries - 1:
                app.logger.warning(f"Servidor Zabbix retornou erro {response.status_code}. Tentando novamente...")
                time.sleep(5)
                continue
            response.raise_for_status()
            response_json = response.json()
            if 'result' in response_json:
                return response_json['result']
            elif 'error' in response_json:
                error_details = f"{response_json['error']['message']}: {response_json['error']['data']}"
                app.logger.error(f"ERRO API Zabbix: {error_details}")
                return {'error': 'APIError', 'details': error_details}
            return []
        except requests.exceptions.RequestException as e:
            app.logger.error(f"ERRO DE CONEXÃO: Falha ao conectar com a API do Zabbix: {e}")
            return {'error': 'RequestException', 'details': str(e)}
    return None

def obter_config_e_token_zabbix(app_config, task_id):
    update_status(task_id, "Conectando ao Zabbix...")
    config_zabbix = {
        'ZABBIX_URL': app_config['ZABBIX_URL'],
        'ZABBIX_USER': app_config['ZABBIX_USER'],
        'ZABBIX_PASSWORD': app_config['ZABBIX_PASSWORD'],
        'ZABBIX_TOKEN': app_config['ZABBIX_TOKEN']
    }
    if not all([config_zabbix['ZABBIX_URL'], config_zabbix['ZABBIX_USER'], config_zabbix['ZABBIX_PASSWORD']]):
        return None, "Variáveis de ambiente do Zabbix (URL, USER, PASSWORD) não configuradas."
    if not config_zabbix['ZABBIX_TOKEN']:
        body = {'jsonrpc': '2.0', 'method': 'user.login', 'params': {'username': config_zabbix['ZABBIX_USER'], 'password': config_zabbix['ZABBIX_PASSWORD']}, 'id': 1}
        token_response = fazer_request_zabbix(body, config_zabbix['ZABBIX_URL'])
        if token_response and 'error' not in token_response:
            config_zabbix['ZABBIX_TOKEN'] = token_response
            update_status(task_id, "Conexão bem-sucedida.")
        else:
            details = token_response.get('details', 'N/A') if isinstance(token_response, dict) else 'Erro desconhecido'
            return None, f"Falha no login do Zabbix. Verifique as credenciais. Detalhes: {details}"
    return config_zabbix, None

class ReportGenerator:
    def __init__(self, config, task_id):
        self.config = config
        self.token = config.get('ZABBIX_TOKEN')
        self.url = config.get('ZABBIX_URL')
        self.task_id = task_id
        if not self.token or not self.url:
            raise ValueError("Configuração do Zabbix não encontrada ou token inválido.")

    def _update_status(self, message):
        update_status(self.task_id, message)
        
    def _get_image_base64(self, file_path):
        if not file_path:
            return None
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)
        if not os.path.exists(full_path):
            return None
        with open(full_path, "rb") as img_file:
            return f"data:image;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"

    def _normalize_string(self, s):
        return re.sub(r'\s+', ' ', str(s).replace('\n', ' ').replace('\r', ' ')).strip()

    def obter_eventos(self, object_ids, periodo, id_type='hostids', max_depth=3):
        time_from, time_till = periodo['start'], periodo['end']
        if max_depth <= 0:
            app.logger.error("ERRO: Limite de profundidade de recursão atingido para obter eventos.")
            return None
        params = {'output': 'extend', 'selectHosts': ['hostid'], 'time_from': time_from, 'time_till': time_till, id_type: object_ids, 'sortfield': ["eventid"], 'sortorder': "ASC"}
        body = {'jsonrpc': '2.0', 'method': 'event.get', 'params': params, 'auth': self.token, 'id': 1}
        resposta = fazer_request_zabbix(body, self.url, allow_retry=False)
        if isinstance(resposta, dict) and 'error' in resposta:
            self._update_status("Consulta pesada detectada, quebrando o período...")
            mid_point = time_from + (time_till - time_from) // 2
            periodo1 = {'start': time_from, 'end': mid_point}
            periodo2 = {'start': mid_point + 1, 'end': time_till}
            eventos1 = self.obter_eventos(object_ids, periodo1, id_type, max_depth - 1)
            if eventos1 is None: return None
            eventos2 = self.obter_eventos(object_ids, periodo2, id_type, max_depth - 1)
            if eventos2 is None: return None
            return eventos1 + eventos2
        return resposta

    def obter_eventos_wrapper(self, object_ids, periodo, id_type='objectids'):
        all_events = []
        if not object_ids: return []
        total_ids = len(object_ids)
        for i, single_id in enumerate(object_ids):
            self._update_status(f"Processando eventos para objeto {i+1}/{total_ids}...")
            eventos_id = self.obter_eventos([single_id], periodo, id_type)
            if eventos_id is None:
                app.logger.critical(f"Falha crítica ao coletar eventos para o ID {single_id}. Abortando.")
                return None
            all_events.extend(eventos_id)
        return sorted(all_events, key=lambda x: int(x['clock']))

    def get_host_groups(self):
        body = {'jsonrpc': '2.0', 'method': 'hostgroup.get', 'params': {'output': ['groupid', 'name'], 'monitored_hosts': True}, 'auth': self.token, 'id': 1}
        return fazer_request_zabbix(body, self.url) or []

    def get_hosts(self, groupids):
        self._update_status("Coletando dados de hosts...")
        body = {'jsonrpc': '2.0', 'method': 'host.get', 'params': {'groupids': groupids, 'selectInterfaces': ['ip'], 'output': ['hostid', 'host', 'name']}, 'auth': self.token, 'id': 1}
        resposta = fazer_request_zabbix(body, self.url)
        if not resposta or 'error' in resposta: return []
        return sorted([{'hostid': item['hostid'], 'hostname': item['host'], 'nome_visivel': self._normalize_string(item['name']), 'ip0': item['interfaces'][0].get('ip', 'N/A') if item.get('interfaces') else 'N/A'} for item in resposta], key=lambda x: x['nome_visivel'])

    def get_items(self, hostids, filter_key):
        self._update_status(f"Buscando itens do tipo '{filter_key}'...")
        body = {'jsonrpc': '2.0', 'method': 'item.get', 'params': {'output': 'extend', 'hostids': hostids, 'selectTriggers': 'extend', 'search': {'key_': filter_key}, 'sortfield': 'name'}, 'auth': self.token, 'id': 1}
        return fazer_request_zabbix(body, self.url) or []

    @staticmethod
    def _correlate_problems(problems, all_events):
        correlated = []
        resolution_events = {p['eventid']: p for p in all_events if p.get('source') == '0' and p.get('value') == '0'}
        for problem in problems:
            r_eventid = problem.get('r_eventid', '0')
            duration = dt.timedelta(seconds=0)
            if r_eventid != '0' and r_eventid in resolution_events:
                res_event = resolution_events[r_eventid]
                if int(res_event['clock']) >= int(problem['clock']):
                    duration = dt.timedelta(seconds=(int(res_event['clock']) - int(problem['clock'])))
            correlated.append({'hostid': problem.get('hosts')[0].get('hostid') if problem.get('hosts') else None, 'duration_seconds': duration.total_seconds()})
        return correlated

    @staticmethod
    def _calculate_sla(correlated_problems, all_hosts, period):
        period_seconds = period['end'] - period['start']
        if period_seconds <= 0: return []
        sla_by_host = {h['hostid']: {'downtime': 0} for h in all_hosts}
        for problem in correlated_problems:
            if problem['hostid'] in sla_by_host:
                sla_by_host[problem['hostid']]['downtime'] += problem['duration_seconds']
        final_results = []
        for host in all_hosts:
            downtime = sla_by_host.get(host['hostid'], {}).get('downtime', 0)
            sla_percent = max(0, 100.0 - (downtime / period_seconds * 100.0))
            final_results.append({'Host': host['nome_visivel'], 'IP': host['ip0'], 'Tempo Indisponível': str(dt.timedelta(seconds=int(downtime))), 'SLA (%)': sla_percent})
        return final_results

    def _count_problems_by_host(self, problems, all_hosts):
        host_map = {h['hostid']: h['nome_visivel'] for h in all_hosts}
        counts = defaultdict(int)
        for p in problems:
            if p.get('object') == '0' and p.get('hosts'):
                host_name = host_map.get(p['hosts'][0]['hostid'])
                if host_name:
                    counts[(host_name, self._normalize_string(p['name']))] += 1
        if not counts: return pd.DataFrame(columns=['Host', 'Problema', 'Ocorrências'])
        df = pd.DataFrame([{'Host': h, 'Problema': p, 'Ocorrências': o} for (h, p), o in counts.items()])
        return df.sort_values(by='Ocorrências', ascending=False)
    
    def _generate_html_sla_table(self, df, sla_goal):
        html = '<table class="table"><thead><tr><th>Host</th><th>IP</th><th>Tempo Indisponível</th><th>SLA (%)</th></tr></thead><tbody>'
        for _, row in df.iterrows():
            sla_val = row['SLA (%)']
            classe_css = ''
            if sla_val < sla_goal: classe_css = 'sla-critico'
            elif sla_val < 99.9: classe_css = 'sla-atencao'
            html += f"<tr class='{classe_css}'><td>{row['Host']}</td><td>{row['IP']}</td><td>{row['Tempo Indisponível']}</td><td>{f'{sla_val:.2f}'.replace('.', ',')}</td></tr>"
        return html + '</tbody></table>'

    def _generate_chart(self, df, x_col, y_col, title, x_label, chart_color):
        self._update_status(f"Gerando gráfico: {title}...")
        if df.empty: return None
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        df_sorted = df.sort_values(by=x_col, ascending=True)
        y_labels = ['\n'.join(textwrap.wrap(str(label), width=50)) for label in df_sorted[y_col]]
        bars = ax.barh(y_labels, df_sorted[x_col], color=chart_color)
        font_size = 8 if len(y_labels) > 10 else 9
        ax.tick_params(axis='y', labelsize=font_size)
        ax.set_xlabel(x_label)
        ax.set_title(title, fontsize=16)
        ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right', 'left', 'bottom']: ax.spines[spine].set_visible(False)
        for bar in bars:
            label_val = bar.get_width()
            label = f'{label_val:.2f}h' if isinstance(label_val, float) else f'{int(label_val)}'
            ax.text(label_val, bar.get_y() + bar.get_height()/2, f' {label}', va='center', ha='left', fontsize=font_size - 1)
        plt.subplots_adjust(left=0.45, right=0.95, top=0.9, bottom=0.1)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, transparent=True)
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def generate(self, client, ref_month_str, system_config, author):
        sla_goal = client.sla_contract
        try:
            ref_date = dt.datetime.strptime(f'{ref_month_str}-01', '%Y-%m-%d')
        except ValueError:
            return None, "Formato de mês de referência inválido. Use YYYY-MM."

        start_date = ref_date.replace(day=1, hour=0, minute=0, second=0)
        end_date = (start_date.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(seconds=1)
        period = {'start': int(start_date.timestamp()), 'end': int(end_date.timestamp())}
        
        # --- COLETA E ANÁLISE DE DADOS ---
        self._update_status("Coletando hosts do cliente...")
        all_hosts = self.get_hosts([client.zabbix_group_id])
        if not all_hosts: return None, f"Nenhum host encontrado para o grupo Zabbix ID {client.zabbix_group_id} do cliente {client.name}."

        all_host_ids = [h['hostid'] for h in all_hosts]
        
        self._update_status("Coletando eventos de disponibilidade (PING)...")
        ping_items = self.get_items(all_host_ids, 'icmpping')
        ping_trigger_ids = list({t['triggerid'] for item in ping_items for t in item.get('triggers', [])})
        ping_events = self.obter_eventos_wrapper(ping_trigger_ids, period, 'objectids')
        if ping_events is None: return None, "Geração abortada: Falha na coleta de eventos de PING."

        self._update_status("Correlacionando problemas de disponibilidade...")
        ping_problems = [p for p in ping_events if p.get('source') == '0' and p.get('object') == '0' and p.get('value') == '1']
        correlated_ping_problems = self._correlate_problems(ping_problems, ping_events)
        df_sla = pd.DataFrame(self._calculate_sla(correlated_ping_problems, all_hosts, period))

        self._update_status("Coletando todos eventos de problemas...")
        all_group_events = self.obter_eventos_wrapper(all_host_ids, period, 'hostids')
        if all_group_events is None: return None, "Geração abortada: Falha na coleta de eventos gerais do grupo."
        
        self._update_status("Analisando dados e gerando estatísticas...")
        all_problems = [p for p in all_group_events if p.get('source') == '0' and p.get('object') == '0' and p.get('value') == '1']
        df_top_incidents = self._count_problems_by_host(all_problems, all_hosts).head(10)
        
        df_sla_problems = df_sla[df_sla['SLA (%)'] < 100.0].copy()
        df_top_downtime = df_sla_problems.sort_values(by='SLA (%)', ascending=True).head(10)
        df_top_downtime['soma_duracao_segundos'] = df_top_downtime['Tempo Indisponível'].apply(lambda x: pd.to_timedelta(x).total_seconds())
        df_top_downtime['soma_duracao_horas'] = df_top_downtime['soma_duracao_segundos'] / 3600
        avg_sla = df_sla['SLA (%)'].mean() if not df_sla.empty else 100.0
        
        principal_ofensor = df_top_incidents.iloc[0]['Host'] if not df_top_incidents.empty else "Nenhum"
        
        kpis_html = f"""
        <div class="kpi-box-single">
            <div class="kpi-value {'status-atingido' if avg_sla >= sla_goal else 'status-nao-atingido'}">{f"{avg_sla:.2f}".replace('.', ',')}%</div>
            <div class="kpi-label">Média de SLA</div>
            <div class="kpi-sublabel">Meta: {f"{sla_goal:.2f}".replace('.', ',')}% | Status: <span class="{'status-atingido' if avg_sla >= sla_goal else 'status-nao-atingido'}">{"Atingido" if avg_sla >= sla_goal else "Não Atingido"}</span></div>
        </div>
        <div class="kpi-box-single">
            <div class="kpi-value">{df_sla[df_sla['SLA (%)'] < 99.9].shape[0]}</div>
            <div class="kpi-label">Hosts com SLA &lt; 99.9%</div>
        </div>
        <div class="kpi-box-single">
            <div class="kpi-value">{len(all_problems)}</div>
            <div class="kpi-label">Total de Incidentes</div>
        </div>
        <div class="kpi-box-single">
            <div class="kpi-value" style="font-size: 18pt;">{principal_ofensor}</div>
            <div class="kpi-label">Principal Ofensor</div>
        </div>
        """

        dados_miolo = {
            'group_name': client.name, 'periodo_referencia': start_date.strftime('%B de %Y').capitalize(), 'data_emissao': dt.datetime.now().strftime('%d/%m/%Y'),
            'total_hosts': len(all_hosts), 'hosts_com_falha_sla': df_sla_problems.shape[0], 'kpis': kpis_html,
            'tabela_sla_problemas': self._generate_html_sla_table(df_sla_problems, sla_goal),
            'tabela_lista_hosts': pd.DataFrame(all_hosts)[['nome_visivel', 'ip0']].rename(columns={'nome_visivel': 'Host', 'ip0': 'IP'}).to_html(classes='table', index=False, border=0),
            'grafico_top_hosts': self._generate_chart(df_top_downtime, 'soma_duracao_horas', 'Host', 'Top 10 Hosts com Maior Indisponibilidade', 'Total de Horas Indisponível', system_config.secondary_color),
            'grafico_top_problemas': self._generate_chart(df_top_incidents.assign(Incidente=df_top_incidents['Host'] + ' - ' + df_top_incidents['Problema']), 'Ocorrências', 'Incidente', 'Top 10 Incidentes', 'Número de Ocorrências', system_config.secondary_color)}
        
        # --- GERAÇÃO DO PDF DE CONTEÚDO ---
        self._update_status("Gerando PDF do conteúdo principal...")
        miolo_html = app.jinja_env.get_template('_MIOLO').render(dados_miolo)
        miolo_pdf_path = os.path.join(app.config['GENERATED_REPORTS_FOLDER'], f"temp_miolo_{self.task_id}.pdf")
        with open(miolo_pdf_path, "w+b") as pdf_file:
            pisa_status = pisa.CreatePDF(BytesIO(miolo_html.encode('UTF-8')), dest=pdf_file)
            if pisa_status.err: return None, f"Falha ao gerar PDF do conteúdo: {pisa_status.err}"
        
        # --- MONTAGEM DO PDF FINAL (LÓGICA SIMPLIFICADA) ---
        self._update_status("Montando o relatório final...")
        merger = PdfWriter()
        
        # 1. Adiciona a Capa (se existir)
        if system_config.report_cover_path:
            cover_path = os.path.join(app.config['UPLOAD_FOLDER'], system_config.report_cover_path)
            if os.path.exists(cover_path):
                try:
                    self._update_status("Adicionando template da capa...")
                    with open(cover_path, "rb") as f_cover:
                        cover_pdf = PdfReader(f_cover)
                        for page in cover_pdf.pages:
                            merger.add_page(page)
                except PyPDF2Errors.PdfReadError:
                    return None, f"Falha ao ler o template da capa ('{system_config.report_cover_path}'). O arquivo pode estar corrompido ou não é um PDF válido."

        # 2. Adiciona o Miolo
        try:
            self._update_status("Adicionando conteúdo principal...")
            with open(miolo_pdf_path, "rb") as f:
                miolo_pdf = PdfReader(f)
                for page in miolo_pdf.pages: merger.add_page(page)
        except PyPDF2Errors.PdfReadError:
             return None, "Ocorreu um erro interno ao gerar o corpo do relatório. O PDF temporário está corrompido."

        # 3. Adiciona a Página Final (se existir)
        if system_config.report_final_page_path:
            final_page_path = os.path.join(app.config['UPLOAD_FOLDER'], system_config.report_final_page_path)
            if os.path.exists(final_page_path):
                try:
                    self._update_status("Adicionando página final...")
                    with open(final_page_path, "rb") as f:
                        final_pdf = PdfReader(f)
                        for page in final_pdf.pages: merger.add_page(page)
                except PyPDF2Errors.PdfReadError:
                    return None, f"Falha ao ler o template da página final ('{system_config.report_final_page_path}'). O arquivo pode estar corrompido ou não é um PDF válido."

        pdf_filename = f'Relatorio_{client.name.replace(" ", "_")}_{ref_month_str}_{uuid.uuid4().hex[:8]}.pdf'
        pdf_path = os.path.join(app.config['GENERATED_REPORTS_FOLDER'], pdf_filename)
        
        with open(pdf_path, "wb") as f:
            merger.write(f)

        # Limpeza
        try:
            os.remove(miolo_pdf_path)
        except OSError as e:
            app.logger.warning(f"Não foi possível remover arquivo temporário: {e}")

        report_record = Report(filename=pdf_filename, file_path=pdf_path, reference_month=ref_month_str, user_id=author.id, client_id=client.id)
        db.session.add(report_record)
        db.session.commit()
        
        AuditService.log(f"Gerou relatório para o cliente '{client.name}' referente a {ref_month_str}", user=author)
        return pdf_path, None

# --- Hooks e Funções de Suporte da Aplicação ---
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.before_request
def before_request_func():
    g.sys_config = SystemConfig.query.first()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not (current_user.has_role('admin') or current_user.has_role('super_admin')):
            flash('Acesso negado. Você não tem permissão para acessar esta área.', 'danger')
            return redirect(url_for('gerar_form'))
        return f(*args, **kwargs)
    return decorated_function

def get_text_color_for_bg(hex_color):
    try:
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return '#212529' if luminance > 0.6 else '#ffffff'
    except (ValueError, TypeError):
        return '#ffffff'

app.jinja_env.filters['text_color_for_bg'] = get_text_color_for_bg

# --- Templates HTML e Configuração do Jinja ---
_LOGIN_HTML_CONTENT = """
<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Login - {{ g.sys_config.company_name }}</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
<style>
:root { --primary-color: {{ g.sys_config.primary_color | default('#2c3e50') }}; --secondary-color: {{ g.sys_config.secondary_color | default('#3498db') }}; }
body { font-family: 'Poppins', sans-serif; background-color: #f4f7f6; min-height: 100vh; }
.login-wrapper { display: flex; min-height: 100vh; }
.login-form-section { flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 2rem; background-color: #ffffff; }
.login-branding-section {
    flex: 1;
    display: none;
    justify-content: center;
    align-items: center;
    background-color: {{ g.sys_config.login_media_bg_color | default('#2c3e50') }};
    color: white;
    text-align: center;
    padding: 2rem;
    position: relative;
    overflow: hidden;
}
.branding-media {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: {{ g.sys_config.login_media_fill_mode | default('cover') }};
    z-index: 1;
}
.branding-content { position: relative; z-index: 2; background-color: rgba(0,0,0,0.2); padding: 1rem; border-radius: .5rem;}
.branding-content h1 { font-family: 'Playfair Display', serif; font-size: 3.5rem; font-weight: 700; line-height: 1.2; }
.branding-content span { font-family: 'Poppins', sans-serif; font-weight: 300; font-style: italic; }
.form-container { width: 100%; max-width: 400px; }
.form-container .logo { max-height: 80px; margin-bottom: 1.5rem; }
.form-container h2 { font-weight: 600; margin-bottom: 0.5rem; }
.form-container .form-control { border-radius: 0.5rem; padding: 0.9rem; border: 1px solid #e0e0e0; }
.form-container .form-control:focus { border-color: var(--secondary-color); box-shadow: 0 0 0 0.25rem rgba(52, 152, 219, 0.25); }
.form-container .btn-primary { background-color: var(--secondary-color); border-color: var(--secondary-color); padding: 0.9rem; font-weight: 600; border-radius: 0.5rem; }
.footer-text { position: absolute; bottom: 1rem; font-size: 0.8rem; color: #999; }
@media (min-width: 992px) { .login-branding-section { display: flex; } }
</style></head>
<body><div class="login-wrapper">
<div class="login-form-section"><div class="form-container">
{% set logo_to_use = g.sys_config.logo_light_bg_path or g.sys_config.logo_dark_bg_path %}
{% if logo_to_use %}<img src="{{ url_for('uploaded_file', filename=logo_to_use) }}" alt="Logo" class="logo">{% endif %}
<h2>Bem-vindo de volta</h2><p class="text-muted mb-4">Faça login para continuar</p>
{% with messages = get_flashed_messages(with_categories=true) %}{% if messages %}{% for category, message in messages %}
<div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">{{ message }}<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>
{% endfor %}{% endif %}{% endwith %}
<form method="POST"><div class="mb-3"><label for="username" class="form-label">Usuário</label><input type="text" class="form-control" id="username" name="username" required></div>
<div class="mb-4"><label for="password" class="form-label">Senha</label><input type="password" class="form-control" id="password" name="password" required></div>
<div class="d-grid"><button type="submit" class="btn btn-primary">Entrar</button></div></form></div>
<p class="footer-text">{{ g.sys_config.footer_text|safe }}</p></div>
<div class="login-branding-section">
    {% if g.sys_config.login_media_path %}
        <img src="{{ url_for('uploaded_file', filename=g.sys_config.login_media_path) }}" class="branding-media" alt="Branding Media">
    {% else %}
        <div class="branding-content"><h1>criamos<br>ambientes<br>de <span>trabalho</span><br>+ felizes</h1></div>
    {% endif %}
</div>
</div><script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script></body></html>
"""

_BASE_TEMPLATE_CONTENT = """
<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{{ title }} - {{ g.sys_config.company_name }}</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"><link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet"><link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
{% set sidebar_text_color = g.sys_config.primary_color | text_color_for_bg %}
:root { --primary-color: {{ g.sys_config.primary_color | default('#2c3e50') }}; --secondary-color: {{ g.sys_config.secondary_color | default('#3498db') }}; --bg-light: #f4f6f9; --sidebar-width: 250px; }
html, body { height: 100%; }
body { font-family: 'Poppins', sans-serif; background-color: var(--bg-light); display: flex; flex-direction: column; }
.wrapper { display: flex; flex-grow: 1; }
.sidebar { width: var(--sidebar-width); min-height: 100vh; background: var(--primary-color); transition: margin-left 0.3s; display: flex; flex-direction: column; flex-shrink: 0; }
#content { flex-grow: 1; padding: 2rem; display: flex; flex-direction: column; }
.main-content { flex-grow: 1; }
.sidebar-header { padding: 1.5rem; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.1); }
.sidebar-header .logo { display: block; margin: 0 auto; }
.sidebar-header h5 { color: {{ sidebar_text_color }}; font-weight: 600; margin-top: 1rem; }
.sidebar .nav-link { color: {{ sidebar_text_color }}; opacity: 0.8; padding: 0.75rem 1.5rem; display: flex; align-items: center; font-weight: 500; border-left: 3px solid transparent; transition: all 0.2s ease; }
.sidebar .nav-link i { margin-right: 1rem; font-size: 1.2rem; }
.sidebar .nav-link:hover { color: {{ sidebar_text_color }}; opacity: 1; background-color: rgba(0,0,0,0.1); }
.sidebar .nav-link.active { color: #fff; opacity: 1; background-color: var(--secondary-color); border-left-color: #fff; font-weight: 600; }
.top-navbar { background-color: #fff; border-radius: 0.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 2rem; }
.top-navbar .nav-link { color: #555; font-weight: 500; }
.top-navbar .nav-link.active { color: var(--secondary-color); font-weight: 600; border-bottom: 2px solid var(--secondary-color); }
.card { border: none; border-radius: 0.75rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
.card-header { background-color: #fff; border-bottom: 1px solid #eee; font-weight: 600; padding: 1rem 1.25rem; }
.footer { text-align: center; padding: 1rem; color: #888; font-size: 0.9em; background-color: #fff; box-shadow: 0 -2px 10px rgba(0,0,0,0.05); margin-top: 2rem; }
.stat-card { color: white; border-radius: 0.75rem; padding: 1.5rem; }
.stat-card h3 { font-size: 2.5rem; font-weight: 700; }
.stat-card p { margin: 0; opacity: 0.9; }
</style></head>
<body><div class="wrapper">
<nav class="sidebar">
<div><div class="sidebar-header">
{% if g.sys_config.logo_dark_bg_path %}
    <img src="{{ url_for('uploaded_file', filename=g.sys_config.logo_dark_bg_path) }}" alt="Logo" class="logo" style="max-height: {{ g.sys_config.logo_size }}px;">
{% endif %}
<h5>{{ g.sys_config.company_name }}</h5></div>
<ul class="nav flex-column">
<li class="nav-item"><a class="nav-link {% if request.endpoint == 'gerar_form' %}active{% endif %}" href="{{ url_for('gerar_form') }}"><i class="bi bi-file-earmark-bar-graph"></i> Gerar Relatório</a></li>
<li class="nav-item"><a class="nav-link {% if request.endpoint == 'history' %}active{% endif %}" href="{{ url_for('history') }}"><i class="bi bi-clock-history"></i> Histórico</a></li>
{% if not current_user.has_role('client') %}
<li class="nav-item"><a class="nav-link {% if 'admin' in request.endpoint %}active{% endif %}" href="{{ url_for('admin_dashboard') }}"><i class="bi bi-gear"></i> Administração</a></li>
{% endif %}
</ul></div>
<div class="mt-auto p-3"><a href="{{ url_for('logout') }}" class="btn btn-outline-light w-100"><i class="bi bi-box-arrow-right"></i> Sair ({{ current_user.username }})</a></div>
</nav>
<main id="content">
<div class="main-content">
{% with messages = get_flashed_messages(with_categories=true) %}{% if messages %}{% for category, message in messages %}
<div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">{{ message }}<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>
{% endfor %}{% endif %}{% endwith %}
{% block content %}{% endblock %}
</div>
<footer class="footer">{{ g.sys_config.footer_text|safe }}</footer>
</main></div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
{% block scripts %}{% endblock %}
</body></html>
"""

_DASHBOARD_HTML_CONTENT = """
{% extends 'admin/base.html' %}
{% block title %}Dashboard{% endblock %}
{% block admin_content %}
<div class="row">
    <div class="col-md-6 col-xl-3 mb-4">
        <div class="stat-card" style="background-color: #2980b9;">
            <h3>{{ stats.users }}</h3><p>Total de Usuários</p>
        </div>
    </div>
    <div class="col-md-6 col-xl-3 mb-4">
        <div class="stat-card" style="background-color: #27ae60;">
            <h3>{{ stats.clients }}</h3><p>Total de Clientes</p>
        </div>
    </div>
    <div class="col-md-6 col-xl-3 mb-4">
        <div class="stat-card" style="background-color: #f39c12;">
            <h3>{{ stats.reports_total }}</h3><p>Relatórios Gerados</p>
        </div>
    </div>
    <div class="col-md-6 col-xl-3 mb-4">
        <div class="stat-card" style="background-color: #8e44ad;">
            <h3>{{ stats.reports_month }}</h3><p>Relatórios este Mês</p>
        </div>
    </div>
</div>
<div class="row">
    <div class="col-lg-6 mb-4">
        <div class="card">
            <div class="card-header">Últimos Relatórios Gerados</div>
            <div class="card-body">
                <ul class="list-group list-group-flush">
                {% for report in latest_reports %}
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        <div><strong>{{ report.client.name }}</strong><br><small class="text-muted">Por {{ report.author.username }} em {{ report.created_at.strftime('%d/%m/%Y') }}</small></div>
                        <a href="{{ url_for('download_report', report_id=report.id) }}" class="btn btn-sm btn-outline-secondary"><i class="bi bi-download"></i></a>
                    </li>
                {% else %}
                    <li class="list-group-item text-center text-muted">Nenhum relatório gerado ainda.</li>
                {% endfor %}
                </ul>
            </div>
        </div>
    </div>
    <div class="col-lg-6 mb-4">
        <div class="card">
            <div class="card-header">Últimos Usuários Adicionados</div>
            <div class="card-body">
                <ul class="list-group list-group-flush">
                {% for user in latest_users %}
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        {{ user.username }}
                        <span class="badge bg-secondary rounded-pill">{{ user.role.name }}</span>
                    </li>
                {% else %}
                    <li class="list-group-item text-center text-muted">Nenhum usuário recente.</li>
                {% endfor %}
                </ul>
            </div>
        </div>
    </div>
</div>
{% endblock %}
"""

_FORM_HTML_CONTENT = """
{% extends 'base.html' %}
{% block title %}Gerar Relatório{% endblock %}
{% block content %}
<div class="row justify-content-center">
    <div class="col-lg-10">
        <div class="card">
            <div class="card-header"><i class="bi bi-sliders"></i> Parâmetros do Relatório</div>
            <div class="card-body">
                <form id="report-form">
                    <div class="mb-3">
                        <label for="client_id" class="form-label">Cliente</label>
                        <select class="form-select" id="client_id" name="client_id" required>
                        {% for client in clients %}<option value="{{ client.id }}">{{ client.name }}</option>{% else %}
                        <option value="" disabled>Nenhum cliente vinculado a você.</option>{% endfor %}</select>
                    </div>
                    <div class="row">
                        <div class="col-md-12 mb-3">
                            <label for="mes_ref" class="form-label">Mês de Referência</label>
                            <input type="month" class="form-control" id="mes_ref" name="mes_ref" required>
                        </div>
                    </div>
                    <div class="d-grid mt-3">
                        <button type="submit" id="submit-btn" class="btn btn-primary btn-lg">
                            <span class="spinner-border spinner-border-sm d-none" role="status" aria-hidden="true"></span>
                            <span class="btn-text"><i class="bi bi-play-circle"></i> Gerar e Baixar Relatório</span>
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
<div class="modal fade" id="progressModal" tabindex="-1" aria-labelledby="progressModalLabel" aria-hidden="true" data-bs-backdrop="static" data-bs-keyboard="false">
  <div class="modal-dialog modal-dialog-centered">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="progressModalLabel">Gerando Relatório...</h5>
      </div>
      <div class="modal-body">
        <p id="status-message">Iniciando...</p>
        <div class="progress">
          <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 100%" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
      </div>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
document.addEventListener("DOMContentLoaded", function() {
    const today = new Date();
    const lastMonth = new Date(today.getFullYear(), today.getMonth() - 1, 1);
    const year = lastMonth.getFullYear();
    const month = String(lastMonth.getMonth() + 1).padStart(2, '0');
    document.getElementById('mes_ref').value = `${year}-${month}`;
    const form = document.getElementById('report-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnSpinner = submitBtn.querySelector('.spinner-border');
    const progressModal = new bootstrap.Modal(document.getElementById('progressModal'));
    const statusMessage = document.getElementById('status-message');
    let intervalId;
    function resetForm() {
        btnText.innerHTML = '<i class="bi bi-play-circle"></i> Gerar e Baixar Relatório';
        btnSpinner.classList.add('d-none');
        submitBtn.disabled = false;
        if (intervalId) clearInterval(intervalId);
        progressModal.hide();
    }
    async function checkStatus(taskId) {
        try {
            const response = await fetch(`/report_status/${taskId}`);
            if (!response.ok) {
                throw new Error('A resposta do servidor não foi OK');
            }
            const data = await response.json();
            statusMessage.textContent = data.status || 'Processando...';
            if (data.status.startsWith('Erro')) {
                clearInterval(intervalId);
                alert(data.status);
                resetForm();
            } else if (data.status === 'Concluído') {
                clearInterval(intervalId);
                statusMessage.textContent = 'Sucesso! Iniciando o download...';
                window.location.href = `/download_final_report/${taskId}`;
                setTimeout(resetForm, 3000);
            }
        } catch (error) {
            console.error('Erro ao verificar status:', error);
            statusMessage.textContent = 'Erro de comunicação ao verificar o status.';
            clearInterval(intervalId);
            setTimeout(resetForm, 2000);
        }
    }
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        btnText.textContent = 'Gerando...';
        btnSpinner.classList.remove('d-none');
        submitBtn.disabled = true;
        const formData = new FormData(form);
        try {
            const response = await fetch("{{ url_for('gerar_relatorio') }}", {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Erro do Servidor: ${errorText}`);
            }
            const data = await response.json();
            if (data.task_id) {
                progressModal.show();
                intervalId = setInterval(() => checkStatus(data.task_id), 1500);
            } else {
                alert(data.error || 'Ocorreu um erro desconhecido ao iniciar a tarefa.');
                resetForm();
            }
        } catch (error) {
            console.error('Erro ao submeter formulário:', error);
            alert('Falha ao iniciar a geração do relatório.');
            resetForm();
        }
    });
});
</script>
{% endblock %}
"""

_HISTORY_HTML_CONTENT = """
{% extends 'base.html' %}
{% block title %}Histórico de Relatórios{% endblock %}
{% block content %}
<div class="card">
    <div class="card-header">Histórico de Relatórios</div>
    <div class="card-body">
        <div class="table-responsive">
            <table class="table table-hover align-middle">
            <thead><tr><th>Cliente</th><th>Mês Ref.</th><th>Gerado Por</th><th>Data</th><th>Ações</th></tr></thead>
            <tbody>
            {% for report in reports %}
            <tr><td>{{ report.client.name }}</td><td>{{ report.reference_month }}</td><td>{{ report.author.username }}</td>
            <td>{{ report.created_at.strftime('%d/%m/%Y %H:%M') }}</td>
            <td><a href="{{ url_for('download_report', report_id=report.id) }}" class="btn btn-sm btn-outline-primary" title="Baixar Novamente"><i class="bi bi-download"></i></a></td></tr>
            {% else %}
            <tr><td colspan="5" class="text-center text-muted p-4">Nenhum relatório no histórico.</td></tr>
            {% endfor %}</tbody></table>
        </div>
    </div>
</div>
{% endblock %}
"""

_ADMIN_BASE_CONTENT = """
{% extends 'base.html' %}
{% block content %}
<nav class="navbar navbar-expand-lg top-navbar">
<div class="container-fluid">
    <ul class="navbar-nav">
        <li class="nav-item"><a class="nav-link {% if request.endpoint == 'admin_dashboard' %}active{% endif %}" href="{{ url_for('admin_dashboard') }}">Dashboard</a></li>
        <li class="nav-item"><a class="nav-link {% if request.endpoint == 'admin_users' or 'user_form' in request.endpoint %}active{% endif %}" href="{{ url_for('admin_users') }}">Usuários</a></li>
        <li class="nav-item"><a class="nav-link {% if request.endpoint == 'admin_clients' or 'client_form' in request.endpoint %}active{% endif %}" href="{{ url_for('admin_clients') }}">Clientes</a></li>
        <li class="nav-item"><a class="nav-link {% if request.endpoint == 'admin_vincular' %}active{% endif %}" href="{{ url_for('admin_vincular') }}">Vínculos</a></li>
        <li class="nav-item"><a class="nav-link {% if request.endpoint == 'admin_customize' %}active{% endif %}" href="{{ url_for('admin_customize') }}">Configurações</a></li>
        <li class="nav-item"><a class="nav-link {% if request.endpoint == 'admin_audit_log' %}active{% endif %}" href="{{ url_for('admin_audit_log') }}">Auditoria</a></li>
    </ul>
</div>
</nav>
{% block admin_content %}{% endblock %}
{% endblock %}
"""

_ADMIN_USERS_CONTENT = """
{% extends 'admin/base.html' %}
{% block title %}Gerenciar Usuários{% endblock %}
{% block admin_content %}
<div class="d-flex justify-content-between align-items-center mb-4">
    <h1 class="h3 mb-0">Gerenciar Usuários</h1>
    <a href="{{ url_for('admin_add_user') }}" class="btn btn-primary"><i class="bi bi-plus-circle"></i> Novo Usuário</a>
</div>
<div class="card"><div class="card-body"><div class="table-responsive">
<table class="table table-hover align-middle">
<thead><tr><th>Usuário</th><th>Função</th><th>Ações</th></tr></thead>
<tbody>
{% for user in users %}
<tr><td>{{ user.username }}</td>
<td>{% if user.has_role('super_admin') %}<span class="badge bg-danger">Super Admin</span>
{% elif user.has_role('admin') %}<span class="badge bg-warning text-dark">Admin</span>
{% else %}<span class="badge bg-info">Cliente</span>{% endif %}</td>
<td>
<a href="{{ url_for('admin_edit_user', user_id=user.id) }}" class="btn btn-sm btn-outline-primary" title="Editar"><i class="bi bi-pencil"></i></a>
{% if user.id != current_user.id %}
<form method="POST" action="{{ url_for('admin_delete_user', user_id=user.id) }}" class="d-inline" onsubmit="return confirm('Tem certeza que deseja excluir o usuário {{ user.username }}?');">
<button type="submit" class="btn btn-sm btn-outline-danger" title="Excluir"><i class="bi bi-trash"></i></button></form>
{% endif %}</td></tr>
{% endfor %}</tbody></table></div></div></div>
{% endblock %}
"""

_ADMIN_USER_FORM_CONTENT = """
{% extends 'admin/base.html' %}
{% block title %}{{ 'Adicionar' if not user else 'Editar' }} Usuário{% endblock %}
{% block admin_content %}
<div class="row justify-content-center"><div class="col-lg-8"><div class="card">
<div class="card-header">{{ 'Adicionar Novo Usuário' if not user else 'Editar Usuário' }}</div>
<div class="card-body"><form method="POST">
<div class="mb-3"><label for="username" class="form-label">Usuário</label>
<input type="text" class="form-control" name="username" value="{{ user.username if user else '' }}" required></div>
<div class="mb-3"><label for="password" class="form-label">Senha</label>
<input type="password" class="form-control" name="password" {% if not user %}required{% endif %}>
<div class="form-text">Deixe em branco para não alterar a senha existente.</div></div>
<div class="mb-3"><label for="role_id" class="form-label">Função</label>
<select class="form-select" name="role_id" required>
{% for role in roles %}<option value="{{ role.id }}" {% if user and user.role_id == role.id %}selected{% endif %}>{{ role.name }}</option>{% endfor %}</select></div>
<div class="mt-4"><button type="submit" class="btn btn-primary">Salvar</button>
<a href="{{ url_for('admin_users') }}" class="btn btn-secondary">Cancelar</a></div></form></div></div></div></div>
{% endblock %}
"""

_ADMIN_CLIENTS_CONTENT = """
{% extends 'admin/base.html' %}
{% block title %}Gerenciar Clientes{% endblock %}
{% block admin_content %}
<div class="d-flex justify-content-between align-items-center mb-4"><h1 class="h3 mb-0">Gerenciar Clientes</h1>
<a href="{{ url_for('admin_add_client') }}" class="btn btn-primary"><i class="bi bi-plus-circle"></i> Novo Cliente</a></div>
<div class="card"><div class="card-body"><div class="table-responsive">
<table class="table table-hover align-middle">
<thead><tr><th></th><th>Nome</th><th>Zabbix Group ID</th><th>SLA Contratado (%)</th><th>Ações</th></tr></thead>
<tbody>
{% for client in clients %}
<tr>
<td>{% if client.logo_path %}<img src="{{ url_for('uploaded_file', filename=client.logo_path) }}" height="30" class="rounded border p-1">{% else %}<span class="text-muted small">Sem logo</span>{% endif %}</td>
<td><strong>{{ client.name }}</strong></td><td>{{ client.zabbix_group_id }}</td><td>{{ client.sla_contract }}</td>
<td>
<a href="{{ url_for('admin_edit_client', client_id=client.id) }}" class="btn btn-sm btn-outline-primary"><i class="bi bi-pencil"></i></a>
<form method="POST" action="{{ url_for('admin_delete_client', client_id=client.id) }}" class="d-inline" onsubmit="return confirm('Tem certeza?');">
<button type="submit" class="btn btn-sm btn-outline-danger"><i class="bi bi-trash"></i></button></form></td></tr>
{% endfor %}</tbody></table></div></div></div>
{% endblock %}
"""

_ADMIN_CLIENT_FORM_CONTENT = """
{% extends 'admin/base.html' %}
{% block title %}{{ 'Adicionar' if not client else 'Editar' }} Cliente{% endblock %}
{% block admin_content %}
<div class="row justify-content-center"><div class="col-lg-8"><div class="card">
<div class="card-header">{{ 'Adicionar Cliente' if not client else 'Editar Cliente' }}</div>
<div class="card-body"><form method="POST" enctype="multipart/form-data">
<div class="mb-3"><label class="form-label">Nome do Cliente</label>
<input type="text" class="form-control" name="name" value="{{ client.name if client else '' }}" required></div>
<div class="mb-3"><label class="form-label">Grupo Zabbix</label>
<select class="form-select" name="zabbix_group_id" required><option value="">Selecione um Grupo</option>
{% for group in zabbix_groups %}<option value="{{ group.groupid }}" {% if client and client.zabbix_group_id == group.groupid %}selected{% endif %}>{{ group.name }}</option>{% endfor %}</select></div>
<div class="mb-3"><label class="form-label">SLA Contratado (%)</label>
<input type="number" step="0.01" class="form-control" name="sla_contract" value="{{ client.sla_contract if client else '99.9' }}" required></div>
<div class="mb-3"><label class="form-label">Logo do Cliente</label><input type="file" class="form-control" name="logo" accept=".png,.jpg,.jpeg">
{% if client and client.logo_path %}<small class="form-text text-muted">Atual: <img src="{{ url_for('uploaded_file', filename=client.logo_path) }}" height="30" class="mt-1 border p-1 rounded"></small>{% endif %}</div>
<button type="submit" class="btn btn-primary">Salvar</button>
<a href="{{ url_for('admin_clients') }}" class="btn btn-secondary">Cancelar</a></form></div></div></div></div>
{% endblock %}
"""

_ADMIN_VINCULAR_CONTENT = """
{% extends 'admin/base.html' %}
{% block title %}Vincular Usuários a Clientes{% endblock %}
{% block admin_content %}
<div class="row justify-content-center"><div class="col-lg-8"><div class="card">
<div class="card-header">Vincular Usuários a Clientes</div>
<div class="card-body"><form method="POST">
    <div class="mb-3">
        <label for="user_id" class="form-label">Usuário (tipo Cliente)</label>
        <select class="form-select" name="user_id" id="user_id" required>
            <option value="">Selecione um usuário</option>
            {% for user in users %}<option value="{{ user.id }}">{{ user.username }}</option>{% endfor %}
        </select>
    </div>
    <div class="mb-3">
        <label class="form-label">Clientes Vinculados</label>
        <div id="client-checkboxes" class="border p-3 rounded" style="max-height: 200px; overflow-y: auto;">
            <p class="text-muted">Selecione um usuário para ver os clientes.</p>
        </div>
    </div>
    <button type="submit" class="btn btn-primary">Salvar Vínculos</button>
</form></div></div></div></div>
<script>
document.getElementById('user_id').addEventListener('change', function() {
    const userId = this.value;
    const checkboxContainer = document.getElementById('client-checkboxes');
    if (!userId) {
        checkboxContainer.innerHTML = '<p class="text-muted">Selecione um usuário para ver os clientes.</p>';
        return;
    }
    fetch(`/admin/get_user_clients/${userId}`)
        .then(response => response.json())
        .then(data => {
            let checkboxesHtml = '';
            data.all_clients.forEach(client => {
                const isChecked = data.linked_clients.includes(client.id);
                checkboxesHtml += `
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" name="client_ids" value="${client.id}" id="client-${client.id}" ${isChecked ? 'checked' : ''}>
                        <label class="form-check-label" for="client-${client.id}">${client.name}</label>
                    </div>
                `;
            });
            checkboxContainer.innerHTML = checkboxesHtml;
        });
});
</script>
{% endblock %}
"""

_ADMIN_CUSTOMIZE_CONTENT = """
{% extends 'admin/base.html' %}
{% block title %}Configurações do Sistema{% endblock %}
{% block admin_content %}
<div class="row justify-content-center"><div class="col-lg-10">
<div class="card mb-4"><div class="card-header">Customização da Aparência</div>
<div class="card-body"><form method="POST" action="{{ url_for('admin_customize_save') }}" enctype="multipart/form-data">
<div class="mb-3"><label class="form-label">Nome da Empresa</label><input type="text" class="form-control" name="company_name" value="{{ g.sys_config.company_name }}" required></div>
<div class="row">
    <div class="col-md-6 mb-3"><label class="form-label">Logo para Fundos Escuros (Sidebar)</label><input type="file" class="form-control" name="logo_dark" accept=".png,.jpg,.jpeg">
    {% if g.sys_config.logo_dark_bg_path %}<small class="form-text text-muted">Atual: <img src="{{ url_for('uploaded_file', filename=g.sys_config.logo_dark_bg_path) }}" height="30" class="mt-1 bg-dark p-1 rounded"></small>{% endif %}</div>
    <div class="col-md-6 mb-3"><label class="form-label">Logo para Fundos Claros (Login)</label><input type="file" class="form-control" name="logo_light" accept=".png,.jpg,.jpeg">
    {% if g.sys_config.logo_light_bg_path %}<small class="form-text text-muted">Atual: <img src="{{ url_for('uploaded_file', filename=g.sys_config.logo_light_bg_path) }}" height="30" class="mt-1 border p-1 rounded"></small>{% endif %}</div>
</div>
<div class="mb-3"><label class="form-label">Tamanho do Logo na Sidebar (Altura em px)</label><input type="number" class="form-control" name="logo_size" value="{{ g.sys_config.logo_size }}"></div>
<div class="row"><div class="col-md-6 mb-3"><label class="form-label">Cor Primária (Sidebar)</label><input type="color" class="form-control form-control-color" name="primary_color" value="{{ g.sys_config.primary_color }}"></div>
<div class="col-md-6 mb-3"><label class="form-label">Cor Secundária (Destaques)</label><input type="color" class="form-control form-control-color" name="secondary_color" value="{{ g.sys_config.secondary_color }}"></div></div>
<div class="mb-3"><label class="form-label">Texto do Rodapé</label><input type="text" class="form-control" name="footer_text" value="{{ g.sys_config.footer_text|striptags }}"></div>
<hr class="my-4">
<h5 class="mb-3">Customização da Tela de Login</h5>
<div class="mb-3"><label class="form-label">Mídia da Tela de Login (Imagem/GIF)</label><input type="file" class="form-control" name="login_media" accept=".png,.jpg,.jpeg,.gif">
{% if g.sys_config.login_media_path %}<small class="form-text text-muted">Atual: <img src="{{ url_for('uploaded_file', filename=g.sys_config.login_media_path) }}" height="50" class="mt-1 border p-1 rounded"></small>{% endif %}</div>
<div class="row">
    <div class="col-md-6 mb-3"><label class="form-label">Cor de Fundo da Mídia</label><input type="color" class="form-control form-control-color" name="login_media_bg_color" value="{{ g.sys_config.login_media_bg_color }}"></div>
    <div class="col-md-6 mb-3"><label class="form-label">Modo de Preenchimento da Mídia</label>
        <select name="login_media_fill_mode" class="form-select">
            <option value="cover" {% if g.sys_config.login_media_fill_mode == 'cover' %}selected{% endif %}>Cobrir (Preenche o espaço, pode cortar)</option>
            <option value="contain" {% if g.sys_config.login_media_fill_mode == 'contain' %}selected{% endif %}>Conter (Mostra imagem inteira, pode ter bordas)</option>
            <option value="fill" {% if g.sys_config.login_media_fill_mode == 'fill' %}selected{% endif %}>Preencher (Estica para caber, pode distorcer)</option>
        </select>
    </div>
</div>
<hr class="my-4">
<h5 class="mb-3">Customização dos Relatórios PDF</h5>
<div class="row">
    <div class="col-md-6 mb-3"><label class="form-label">Template da Página de Capa (PDF)</label><input type="file" class="form-control" name="report_cover" accept=".pdf">
    {% if g.sys_config.report_cover_path %}<small class="form-text text-muted">Atual: {{ g.sys_config.report_cover_path }}</small>{% endif %}</div>
    <div class="col-md-6 mb-3"><label class="form-label">Template da Página Final (PDF)</label><input type="file" class="form-control" name="report_final_page" accept=".pdf">
    {% if g.sys_config.report_final_page_path %}<small class="form-text text-muted">Atual: {{ g.sys_config.report_final_page_path }}</small>{% endif %}</div>
</div>
<button type="submit" class="btn btn-primary mt-3">Salvar Aparência</button></form></div></div>
<div class="card"><div class="card-header">Diagnóstico</div>
<div class="card-body"><p>Verifique se as credenciais do Zabbix no arquivo <code>.env</code> estão corretas.</p>
<form method="POST" action="{{ url_for('admin_test_zabbix') }}"><button type="submit" class="btn btn-info">Testar Conexão com Zabbix</button></form></div></div></div></div>
{% endblock %}
"""

_ADMIN_AUDIT_LOG_CONTENT = """
{% extends 'admin/base.html' %}
{% block title %}Log de Auditoria{% endblock %}
{% block admin_content %}
<div class="card"><div class="card-header">Log de Auditoria</div>
<div class="card-body"><div class="table-responsive">
<table class="table table-sm table-hover">
<thead><tr><th>Data/Hora (UTC)</th><th>Usuário</th><th>Ação</th></tr></thead>
<tbody>
{% for log in logs %}
<tr><td>{{ log.timestamp.strftime('%d/%m/%Y %H:%M:%S') }}</td><td>{{ log.username }}</td><td>{{ log.action }}</td></tr>
{% else %}
<tr><td colspan="3" class="text-center text-muted p-4">Nenhum registro de auditoria encontrado.</td></tr>
{% endfor %}</tbody></table></div></div></div>
{% endblock %}
"""

_TEMPLATE_RELATORIO_MIOLO = """
<!DOCTYPE html><html lang="pt-br"><head><meta charset="UTF-8"><title>Miolo do Relatório</title>
<style>
@page { size: a4 portrait; margin: 2.5cm 1.5cm; @frame header_frame { -pdf-frame-content: header_content; left: 1.5cm; right: 1.5cm; top: 1cm; height: 2cm; } @frame footer_frame { -pdf-frame-content: footer_content; left: 1.5cm; right: 1.5cm; bottom: 1cm; height: 1cm; }}
body { font-family: 'Helvetica', 'Arial', sans-serif; font-size: 10pt; color: #333; }
h1, h2, h3 { font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif; font-weight: 300; color: #0d47a1; }
h1 { font-size: 22pt; text-align: center; margin-bottom: 20px; }
h2 { font-size: 16pt; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 25px; }
h3 { font-size: 13pt; margin-top: 20px; }
thead { display: table-header-group; }
.table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 8pt; }
.table th, .table td { border: 1px solid #e0e0e0; padding: 6px; text-align: left; }
.table th { background-color: #1e88e5; color: #fff; font-weight: bold; }
.chart-container { text-align: center; margin-top: 20px; page-break-inside: avoid; }
.chart-container img { max-width: 95%; height: auto; }
.summary-box { border: 1px solid #90caf9; background-color: #e3f2fd; padding: 15px; margin-top: 10px; margin-bottom: 25px; border-radius: 5px; }
/* [LAYOUT AJUSTADO] Novos estilos de paginação e KPIs */
.kpi-container {
    page-break-inside: avoid; /* Impede que o bloco de KPIs seja quebrado entre páginas */
}
.new-page-section {
    page-break-before: always; /* Força o início de uma nova página para a seção */
}
.chart-new-page {
    page-break-before: always; /* Força o início de uma nova página para o segundo gráfico */
}
.kpi-box-single {
    width: 100%;
    background-color: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 15px;
    text-align: center;
    page-break-inside: avoid;
}
.kpi-value {
    font-size: 28pt;
    font-weight: bold;
    color: #0d47a1; /* Cor azul padrão */
}
.kpi-label {
    font-size: 12pt;
    color: #333;
    margin-top: 5px;
}
.kpi-sublabel {
    font-size: 9pt;
    color: #555;
    margin-top: 5px;
}
.status-atingido {
    color: #2e7d32 !important; /* Verde com !important para garantir a sobreposição */
    font-weight: bold;
}
.status-nao-atingido {
    color: #c62828 !important; /* Vermelho com !important para garantir a sobreposição */
    font-weight: bold;
}
.sla-critico { background-color: #ffebee !important; color: #c62828; font-weight: bold; }
.sla-atencao { background-color: #fffde7 !important; color: #f57f17; }
#header_content, #footer_content { font-size: 9pt; color: #555; }
#footer_content { text-align: right; }
</style></head>
<body>
<div id="header_content">Relatório de Monitoramento | {{ group_name }}</div>
<div id="footer_content">Gerado em {{ data_emissao }} | Página <pdf:pagenumber> de <pdf:pagecount></div>
<h1>Relatório Executivo</h1>
<div class="summary-box"><p><strong>Cliente:</strong> {{ group_name }}</p><p><strong>Período:</strong> {{ periodo_referencia }}</p></div>
<h2>Indicadores Chave (KPIs)</h2>
<div class="kpi-container">
    {{ kpis | safe }}
</div>
<h2 class="new-page-section">1. Análise de Disponibilidade (SLA)</h2>
<div class="summary-box">
    {% if hosts_com_falha_sla == 0 %}<p style="text-align:center; font-weight:bold; color: #2e7d32;">Parabéns! 100% dos {{ total_hosts }} hosts atingiram a meta de disponibilidade.</p>
    {% else %}<p style="text-align:center; font-weight:bold; color: #c62828;">Atenção: {{ hosts_com_falha_sla }} de {{ total_hosts }} hosts não atingiram 100% de disponibilidade.</p>{% endif %}
</div>
{{ tabela_sla_problemas|safe }}
<h2 class="new-page-section">2. Visão Geral de Incidentes</h2>
<h3>2.1. Hosts com Maior Indisponibilidade</h3>
<div class="chart-container">{% if grafico_top_hosts %}<img src="data:image/png;base64,{{ grafico_top_hosts }}">{% else %}<p><i>Nenhum host com indisponibilidade.</i></p>{% endif %}</div>
<h3 class="chart-new-page">2.2. Top 10 Incidentes</h3>
<div class="chart-container">{% if grafico_top_problemas %}<img src="data:image/png;base64,{{ grafico_top_problemas }}">{% else %}<p><i>Nenhum problema registrado.</i></p>{% endif %}</div>
<h2 class="new-page-section">Anexo A: Inventário Completo</h2>
{{ tabela_lista_hosts|safe }}
</body></html>
"""

_JINJA_TEMPLATES = {
    'login.html': _LOGIN_HTML_CONTENT,
    'base.html': _BASE_TEMPLATE_CONTENT,
    'dashboard.html': _DASHBOARD_HTML_CONTENT,
    'gerar_form.html': _FORM_HTML_CONTENT,
    'history.html': _HISTORY_HTML_CONTENT,
    'admin/base.html': _ADMIN_BASE_CONTENT,
    'admin/users.html': _ADMIN_USERS_CONTENT,
    'admin/user_form.html': _ADMIN_USER_FORM_CONTENT,
    'admin/clients.html': _ADMIN_CLIENTS_CONTENT,
    'admin/client_form.html': _ADMIN_CLIENT_FORM_CONTENT,
    'admin/vincular.html': _ADMIN_VINCULAR_CONTENT,
    'admin/customize.html': _ADMIN_CUSTOMIZE_CONTENT,
    'admin/audit_log.html': _ADMIN_AUDIT_LOG_CONTENT,
    '_MIOLO': _TEMPLATE_RELATORIO_MIOLO,
}
app.jinja_env.loader = DictLoader(_JINJA_TEMPLATES)

# --- Rotas da Aplicação (Views) ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: 
        return redirect(url_for('gerar_form'))
            
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            AuditService.log("Login bem-sucedido", user=user)
            return redirect(url_for('gerar_form'))
        else:
            AuditService.log(f"Tentativa de login falhou para o usuário '{request.form['username']}'")
            flash('Usuário ou senha inválidos.', 'danger')
    return render_template_string(_JINJA_TEMPLATES['login.html'])

@app.route('/logout')
@login_required
def logout():
    AuditService.log("Logout realizado")
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return redirect(url_for('gerar_form'))

@app.route('/gerar')
@login_required
def gerar_form():
    if current_user.has_role('client'):
        clients = current_user.clients
    else:
        clients = Client.query.order_by(Client.name).all()
    return render_template_string(_JINJA_TEMPLATES['gerar_form.html'], title="Gerar Relatório", clients=clients)

def run_generation_in_thread(app_context, task_id, client_id, ref_month, user_id):
    with app_context:
        try:
            system_config = SystemConfig.query.first()
            if not system_config:
                update_status(task_id, "Erro: Configuração do sistema não encontrada no banco de dados.")
                return

            author = db.session.get(User, user_id)
            if not author:
                 update_status(task_id, f"Erro: Usuário com ID {user_id} não encontrado.")
                 return

            client = db.session.get(Client, int(client_id))
            
            if not client or (author.has_role('client') and client not in author.clients):
                update_status(task_id, "Erro: Cliente inválido ou não autorizado.")
                return

            config_zabbix, erro_zabbix_config = obter_config_e_token_zabbix(app.config, task_id)
            if erro_zabbix_config:
                update_status(task_id, f"Erro: {erro_zabbix_config}")
                return

            generator = ReportGenerator(config_zabbix, task_id)
            pdf_path, error = generator.generate(client, ref_month, system_config, author)

            if error:
                update_status(task_id, f"Erro: {error}")
            else:
                with TASK_LOCK:
                    REPORT_GENERATION_TASKS[task_id]['file_path'] = pdf_path
                    REPORT_GENERATION_TASKS[task_id]['status'] = "Concluído"

        except Exception as e:
            error_trace = traceback.format_exc()
            app.logger.error(f"Erro fatal na thread de geração de relatório (Task ID: {task_id}):\n{error_trace}")
            update_status(task_id, "Erro: Falha crítica durante a geração. Contate o administrador.")

@app.route('/gerar_relatorio', methods=['POST'])
@login_required
def gerar_relatorio():
    task_id = str(uuid.uuid4())
    with TASK_LOCK:
        REPORT_GENERATION_TASKS[task_id] = {'status': 'Iniciando...'}
    
    client_id = request.form.get('client_id')
    ref_month = request.form.get('mes_ref')

    thread = threading.Thread(target=run_generation_in_thread, args=(app.app_context(), task_id, client_id, ref_month, current_user.id))
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/report_status/<task_id>')
@login_required
def report_status(task_id):
    with TASK_LOCK:
        task = REPORT_GENERATION_TASKS.get(task_id, {'status': 'Tarefa não encontrada.'})
    return jsonify(task)

@app.route('/download_final_report/<task_id>')
@login_required
def download_final_report(task_id):
    with TASK_LOCK:
        task = REPORT_GENERATION_TASKS.get(task_id)
    
    if not task or 'file_path' not in task:
        flash("Arquivo do relatório não encontrado ou a tarefa expirou.", "danger")
        return redirect(url_for('gerar_form'))
    
    file_path = task['file_path']
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash("Arquivo do relatório não existe mais no servidor.", "danger")
        return redirect(url_for('gerar_form'))

@app.route('/history')
@login_required
def history():
    if current_user.has_role('client'):
        client_ids = [c.id for c in current_user.clients]
        query = Report.query.filter(Report.client_id.in_(client_ids))
    else:
        query = Report.query
    reports = query.order_by(Report.created_at.desc()).all()
    return render_template_string(_JINJA_TEMPLATES['history.html'], title="Histórico", reports=reports)

@app.route('/download_report/<int:report_id>')
@login_required
def download_report(report_id):
    report = db.session.get(Report, report_id)
    if not report:
        flash("Relatório não encontrado.", "danger")
        return redirect(url_for('history'))
    
    is_authorized = False
    if not current_user.has_role('client'):
        is_authorized = True
    elif report.client in current_user.clients:
        is_authorized = True

    if not is_authorized:
        flash("Acesso negado.", "danger")
        return redirect(url_for('history'))
        
    AuditService.log(f"Re-download do relatório '{report.filename}'")
    return send_file(report.file_path, as_attachment=True)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Rotas de Administração ---
@app.route('/admin')
@admin_required
def admin_dashboard():
    stats = {
        'users': User.query.count(),
        'clients': Client.query.count(),
        'reports_total': Report.query.count(),
        'reports_month': Report.query.filter(Report.created_at >= dt.datetime.now().replace(day=1)).count()
    }
    latest_reports = Report.query.order_by(Report.created_at.desc()).limit(5).all()
    latest_users = User.query.order_by(User.id.desc()).limit(5).all()
    return render_template_string(_JINJA_TEMPLATES['dashboard.html'], title="Dashboard Admin", stats=stats, latest_reports=latest_reports, latest_users=latest_users)

@app.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.order_by(User.username).all()
    return render_template_string(_JINJA_TEMPLATES['admin/users.html'], title="Gerenciar Usuários", users=users)

@app.route('/admin/users/add', methods=['GET', 'POST'])
@admin_required
def admin_add_user():
    if request.method == 'POST':
        username = request.form['username']
        if User.query.filter_by(username=username).first():
            flash(f"Usuário '{username}' já existe.", "danger")
        else:
            user = User(username=username, role_id=int(request.form['role_id']))
            user.set_password(request.form['password'])
            db.session.add(user)
            db.session.commit()
            AuditService.log(f"Adicionou novo usuário '{username}'")
            flash('Usuário adicionado!', 'success')
            return redirect(url_for('admin_users'))
    roles = Role.query.all()
    return render_template_string(_JINJA_TEMPLATES['admin/user_form.html'], title="Adicionar Usuário", roles=roles, user=None)

@app.route('/admin/users/edit/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_user(user_id):
    user = db.session.get(User, user_id)
    if request.method == 'POST':
        AuditService.log(f"Editou o usuário '{user.username}' (ID: {user_id})")
        user.username = request.form['username']
        if request.form['password']: user.set_password(request.form['password'])
        user.role_id = int(request.form['role_id'])
        db.session.commit()
        flash('Usuário atualizado!', 'success')
        return redirect(url_for('admin_users'))
    roles = Role.query.all()
    return render_template_string(_JINJA_TEMPLATES['admin/user_form.html'], title="Editar Usuário", user=user, roles=roles)

@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    if user_id == current_user.id:
        flash('Você não pode excluir a si mesmo.', 'danger')
        return redirect(url_for('admin_users'))
    user = db.session.get(User, user_id)
    AuditService.log(f"Excluiu o usuário '{user.username}' (ID: {user_id})")
    db.session.delete(user)
    db.session.commit()
    flash('Usuário excluído.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/clients')
@admin_required
def admin_clients():
    clients = Client.query.order_by(Client.name).all()
    return render_template_string(_JINJA_TEMPLATES['admin/clients.html'], title="Gerenciar Clientes", clients=clients)

def save_file_for_model(model_instance, attribute_name, file_key):
    file = request.files.get(file_key)
    if file and allowed_file(file.filename):
        old_path = getattr(model_instance, attribute_name, None)
        if old_path and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], old_path)):
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], old_path))
            except OSError as e:
                app.logger.error(f"Erro ao remover arquivo antigo {old_path}: {e}")
        
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        setattr(model_instance, attribute_name, filename)
        return True
    return False

@app.route('/admin/clients/add', methods=['GET', 'POST'])
@admin_required
def admin_add_client():
    if request.method == 'POST':
        name = request.form['name']
        client = Client(name=name, zabbix_group_id=request.form['zabbix_group_id'], sla_contract=float(request.form['sla_contract']))
        save_file_for_model(client, 'logo_path', 'logo')
        db.session.add(client)
        db.session.commit()
        AuditService.log(f"Adicionou novo cliente '{name}'")
        flash('Cliente adicionado!', 'success')
        return redirect(url_for('admin_clients'))
    
    config_zabbix, erro = obter_config_e_token_zabbix(app.config, 'admin_task')
    zabbix_groups = []
    if not erro:
        gen = ReportGenerator(config_zabbix, 'admin_task')
        zabbix_groups = gen.get_host_groups()
    else:
        flash(f"Aviso: Não foi possível carregar grupos do Zabbix. {erro}", "warning")

    return render_template_string(_JINJA_TEMPLATES['admin/client_form.html'], title="Adicionar Cliente", zabbix_groups=zabbix_groups, client=None)

@app.route('/admin/clients/edit/<int:client_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_client(client_id):
    client = db.session.get(Client, client_id)
    if request.method == 'POST':
        AuditService.log(f"Editou o cliente '{client.name}' (ID: {client_id})")
        client.name = request.form['name']
        client.zabbix_group_id = request.form['zabbix_group_id']
        client.sla_contract = float(request.form['sla_contract'])
        save_file_for_model(client, 'logo_path', 'logo')
        db.session.commit()
        flash('Cliente atualizado!', 'success')
        return redirect(url_for('admin_clients'))
    
    config_zabbix, erro = obter_config_e_token_zabbix(app.config, 'admin_task')
    zabbix_groups = []
    if not erro:
        gen = ReportGenerator(config_zabbix, 'admin_task')
        zabbix_groups = gen.get_host_groups()
    else:
        flash(f"Aviso: Não foi possível carregar grupos do Zabbix. {erro}", "warning")
        
    return render_template_string(_JINJA_TEMPLATES['admin/client_form.html'], title="Editar Cliente", client=client, zabbix_groups=zabbix_groups)

@app.route('/admin/clients/delete/<int:client_id>', methods=['POST'])
@admin_required
def admin_delete_client(client_id):
    client = db.session.get(Client, client_id)
    AuditService.log(f"Excluiu o cliente '{client.name}' (ID: {client_id})")
    db.session.delete(client)
    db.session.commit()
    flash('Cliente excluído.', 'success')
    return redirect(url_for('admin_clients'))

@app.route('/admin/vincular', methods=['GET', 'POST'])
@admin_required
def admin_vincular():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        client_ids = request.form.getlist('client_ids')
        user = db.session.get(User, int(user_id))
        if user:
            user.clients = [db.session.get(Client, int(cid)) for cid in client_ids]
            db.session.commit()
            AuditService.log(f"Atualizou vínculos para o usuário '{user.username}'")
            flash(f"Vínculos para {user.username} atualizados.", "success")
        return redirect(url_for('admin_vincular'))
    
    users = User.query.filter(User.role.has(name='client')).order_by(User.username).all()
    return render_template_string(_JINJA_TEMPLATES['admin/vincular.html'], title="Vínculos", users=users)

@app.route('/admin/get_user_clients/<int:user_id>')
@admin_required
def get_user_clients(user_id):
    user = db.session.get(User, user_id)
    all_clients = Client.query.order_by(Client.name).all()
    linked_client_ids = [c.id for c in user.clients] if user else []
    return {
        "all_clients": [{"id": c.id, "name": c.name} for c in all_clients],
        "linked_clients": linked_client_ids
    }

@app.route('/admin/customize')
@admin_required
def admin_customize():
    return render_template_string(_JINJA_TEMPLATES['admin/customize.html'], title="Configurações")

@app.route('/admin/customize/save', methods=['POST'])
@admin_required
def admin_customize_save():
    AuditService.log("Atualizou as configurações do sistema")
    config = g.sys_config
    config.company_name = request.form['company_name']
    config.footer_text = request.form['footer_text']
    config.primary_color = request.form['primary_color']
    config.secondary_color = request.form['secondary_color']
    config.logo_size = int(request.form.get('logo_size', 50))
    config.login_media_fill_mode = request.form.get('login_media_fill_mode', 'cover')
    config.login_media_bg_color = request.form.get('login_media_bg_color', '#2c3e50')

    save_file_for_model(config, 'logo_dark_bg_path', 'logo_dark')
    save_file_for_model(config, 'logo_light_bg_path', 'logo_light')
    save_file_for_model(config, 'login_media_path', 'login_media')
    save_file_for_model(config, 'report_cover_path', 'report_cover')
    save_file_for_model(config, 'report_final_page_path', 'report_final_page')
    
    db.session.commit()
    flash('Customizações salvas com sucesso!', 'success')
    return redirect(url_for('admin_customize'))

@app.route('/admin/audit')
@admin_required
def admin_audit_log():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(200).all()
    return render_template_string(_JINJA_TEMPLATES['admin/audit_log.html'], title="Log de Auditoria", logs=logs)

@app.route('/admin/test_zabbix', methods=['POST'])
@admin_required
def admin_test_zabbix():
    _, error = obter_config_e_token_zabbix(app.config, 'admin_task')
    if not error:
        flash("Conexão com a API do Zabbix bem-sucedida!", "success")
        AuditService.log("Teste de conexão Zabbix: SUCESSO")
    else:
        flash(f"Falha na conexão com Zabbix: {error}", "danger")
        AuditService.log(f"Teste de conexão Zabbix: FALHA ({error})")
    return redirect(url_for('admin_customize'))

# --- Bloco de Execução Principal ---
if __name__ == '__main__':
    with app.app_context():
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['GENERATED_REPORTS_FOLDER'], exist_ok=True)
        db.create_all()
        
        if not Role.query.first():
            db.session.add_all([Role(name='super_admin'), Role(name='admin'), Role(name='client')])
            db.session.commit()
        if not User.query.filter_by(username='superadmin').first():
            super_admin_role = Role.query.filter_by(name='super_admin').first()
            admin_user = User(username='superadmin', role=super_admin_role)
            admin_user.set_password(app.config['SUPERADMIN_PASSWORD'])
            db.session.add(admin_user)
            db.session.commit()
            print("="*60)
            print(">>> USUÁRIO 'superadmin' CRIADO! <<<")
            if app.config['SUPERADMIN_PASSWORD'] == 'admin123':
                 print(">>> SENHA PADRÃO: 'admin123' <<<")
                 print(">>> MUDE ESTA SENHA EM PRODUÇÃO USANDO A VARIAVEL DE AMBIENTE 'SUPERADMIN_PASSWORD'! <<<")
            else:
                 print(">>> Senha definida pela variável de ambiente 'SUPERADMIN_PASSWORD'.")
            print("="*60)

        if not SystemConfig.query.first():
            db.session.add(SystemConfig())
            db.session.commit()

    app.run(host='0.0.0.0', port=5000, debug=True)