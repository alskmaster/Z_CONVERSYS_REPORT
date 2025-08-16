# -*- coding: utf-8 -*-
# ==============================================================================
# ZABBIX REPORTER - ENTERPRISE EDITION v20.9.6 (KPI as Image)
#
# Autor: Marcio Bernardo, Conversys IT Solutions
# Data: 15/08/2025
# Descrição: Módulo KPI agora é gerado como imagem para garantir design consistente.
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
from flask import (Flask, render_template, request, flash, redirect,
                   url_for, send_file, send_from_directory, get_flashed_messages, g, jsonify, session)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, login_user, logout_user, login_required,
                       current_user, UserMixin)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import requests
import pandas as pd
from jinja2 import TemplateNotFound
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xhtml2pdf import pisa
from PyPDF2 import PdfWriter, PdfReader, errors as PyPDF2Errors

import imgkit

# --- TLS verification controls ---
APP_ENV = os.getenv('APP_ENV', 'prod').lower()
VERIFY_TLS = (os.getenv('VERIFY_TLS', 'true').lower() in ('1','true','yes'))
try:
    import urllib3
    # Only suppress warnings if verification is explicitly disabled
    if not VERIFY_TLS:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass


# --- Carregamento de Configurações ---
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'mude-esta-chave-secreta-em-producao-agora'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///zabbix_reporter_v20.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'uploads'
    GENERATED_REPORTS_FOLDER = 'relatorios_gerados'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
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

class ClientZabbixGroup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    zabbix_group_id = db.Column(db.String(50), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)

class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    sla_contract = db.Column(db.Float, nullable=False, default=99.9)
    logo_path = db.Column(db.String(255), nullable=True)
    reports = db.relationship('Report', backref='client', lazy=True, cascade="all, delete-orphan")
    zabbix_groups = db.relationship('ClientZabbixGroup', backref='client', lazy=True, cascade="all, delete-orphan")

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
    report_type = db.Column(db.String(50), default='custom', nullable=False)

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

def obter_config_e_token_zabbix(app_config, task_id="generic_task"):
    is_threaded_task = task_id != "generic_task"
    if is_threaded_task:
        update_status(task_id, "Conectando ao Zabbix...")
        
    config_zabbix = {
        'ZABBIX_URL': app_config['ZABBIX_URL'],
        'ZABBIX_USER': app_config['ZABBIX_USER'],
        'ZABBIX_PASSWORD': app_config['ZABBIX_PASSWORD'],
        'ZABBIX_TOKEN': app_config['ZABBIX_TOKEN']
    }
    if not all([config_zabbix['ZABBIX_URL'], config_zabbix['ZABBIX_USER'], config_zabbix['ZABBIX_PASSWORD']]):
        return None, "Variáveis de ambiente do Zabbix (URL, USER, PASSWORD) não configuradas."
    
    if not config_zabbix.get('ZABBIX_TOKEN'):
        body = {'jsonrpc': '2.0', 'method': 'user.login', 'params': {'username': config_zabbix['ZABBIX_USER'], 'password': config_zabbix['ZABBIX_PASSWORD']}, 'id': 1}
        token_response = fazer_request_zabbix(body, config_zabbix['ZABBIX_URL'])
        if token_response and 'error' not in token_response:
            config_zabbix['ZABBIX_TOKEN'] = token_response
            if is_threaded_task:
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

    def get_trends(self, itemids, time_from, time_till):
        self._update_status(f"Buscando tendências para {len(itemids)} itens...")
        body = {'jsonrpc': '2.0', 'method': 'trend.get', 'params': {'output': ['itemid', 'clock', 'num', 'value_min', 'value_avg', 'value_max'], 'itemids': itemids, 'time_from': time_from, 'time_till': time_till}, 'auth': self.token, 'id': 1}
        trends = fazer_request_zabbix(body, self.url)
        if not isinstance(trends, list):
            app.logger.error(f"Falha ao buscar trends para {len(itemids)} itens. Resposta inválida do Zabbix.")
            return []
        return trends

    def get_host_groups(self):
        body = {'jsonrpc': '2.0', 'method': 'hostgroup.get', 'params': {'output': ['groupid', 'name'], 'monitored_hosts': True}, 'auth': self.token, 'id': 1}
        return fazer_request_zabbix(body, self.url) or []

    def get_hosts(self, groupids):
        self._update_status("Coletando dados de hosts...")
        body = {'jsonrpc': '2.0', 'method': 'host.get', 'params': {'groupids': groupids, 'selectInterfaces': ['ip'], 'output': ['hostid', 'host', 'name']}, 'auth': self.token, 'id': 1}
        resposta = fazer_request_zabbix(body, self.url)
        if not isinstance(resposta, list): return []
        return sorted([{'hostid': item['hostid'], 'hostname': item['host'], 'nome_visivel': self._normalize_string(item['name']), 'ip0': item['interfaces'][0].get('ip', 'N/A') if item.get('interfaces') else 'N/A'} for item in resposta], key=lambda x: x['nome_visivel'])

    def get_items(self, hostids, filter_key, search_by_key=False):
        self._update_status(f"Buscando itens com filtro '{filter_key}'...")
        params = {'output': ['itemid', 'hostid'], 'hostids': hostids, 'sortfield': 'name'}
        if search_by_key:
            params['search'] = {'key_': filter_key}
            params['selectTriggers'] = 'extend'
        else:
            params['search'] = {'name': filter_key}
        
        body = {'jsonrpc': '2.0', 'method': 'item.get', 'params': params, 'auth': self.token, 'id': 1}
        return fazer_request_zabbix(body, self.url) or []

    def _collect_availability_data(self, all_hosts, period, sla_goal):
        all_host_ids = [h['hostid'] for h in all_hosts]
        
        ping_items = self.get_items(all_host_ids, 'icmpping', search_by_key=True)
        if not ping_items:
            return None, "Nenhum item de monitoramento de PING ('icmpping') encontrado para este grupo."
        
        hosts_with_ping_ids = {item['hostid'] for item in ping_items}
        hosts_for_sla = [host for host in all_hosts if host['hostid'] in hosts_with_ping_ids]
        if not hosts_for_sla:
             return None, "Nenhum dos hosts neste grupo tem um item de PING para calcular o SLA."
        app.logger.info(f"Dos {len(all_hosts)} hosts totais, {len(hosts_for_sla)} possuem itens de PING e serão analisados para SLA.")

        ping_trigger_ids = list({t['triggerid'] for item in ping_items for t in item.get('triggers', [])})
        if not ping_trigger_ids:
            return None, "Nenhum gatilho (trigger) de PING encontrado para os itens deste grupo."

        ping_events = self.obter_eventos_wrapper(ping_trigger_ids, period, 'objectids')
        if ping_events is None: return None, "Geração abortada: Falha na coleta de eventos de PING."

        ping_problems = [p for p in ping_events if p.get('source') == '0' and p.get('object') == '0' and p.get('value') == '1']
        correlated_ping_problems = self._correlate_problems(ping_problems, ping_events)
        df_sla = pd.DataFrame(self._calculate_sla(correlated_ping_problems, hosts_for_sla, period))

        all_group_events = self.obter_eventos_wrapper(all_host_ids, period, 'hostids')
        if all_group_events is None: return None, "Geração abortada: Falha na coleta de eventos gerais do grupo."
        
        all_problems = [p for p in all_group_events if p.get('source') == '0' and p.get('object') == '0' and p.get('value') == '1']
        df_top_incidents = self._count_problems_by_host(all_problems, all_hosts).head(10)
        
        df_sla_problems = df_sla
        avg_sla = df_sla['SLA (%)'].mean() if not df_sla.empty else 100.0
        principal_ofensor = df_top_incidents.iloc[0]['Host'] if not df_top_incidents.empty else "Nenhum"
        
        kpis_data = [
            {
                'label': f"Média de SLA ({len(hosts_for_sla)} Hosts)",
                'value': f"{avg_sla:.2f}".replace('.', ',') + '%',
                'sublabel': f"Meta: {f'{sla_goal:.2f}'.replace('.', ',')}%",
                'status': "atingido" if avg_sla >= sla_goal else "nao-atingido"
            },
            {
                'label': "Hosts com SLA < 99.9%",
                'value': df_sla[df_sla['SLA (%)'] < 99.9].shape[0],
                'sublabel': f"De um total de {len(hosts_for_sla)} hosts",
                'status': "critico" if df_sla[df_sla['SLA (%)'] < 99.9].shape[0] > 0 else "ok"
            },
            {
                'label': "Total de Incidentes",
                'value': len(all_problems),
                'sublabel': "Eventos de problema registrados",
                'status': "info"
            },
            {
                'label': "Principal Ofensor",
                'value': principal_ofensor,
                'sublabel': "Host com mais incidentes",
                'status': "info"
            }
        ]

        return {
            'kpis': kpis_data,
            'df_sla_problems': df_sla_problems,
            'df_top_incidents': df_top_incidents,
        }, None
    
    def _process_trends(self, trends, items, host_map, unit_conversion_factor=1, is_pavailable=False):
        if not isinstance(trends, list) or not trends:
            return pd.DataFrame(columns=['Host', 'Min', 'Max', 'Avg'])

        df = pd.DataFrame(trends)
        df[['value_min', 'value_avg', 'value_max']] = df[['value_min', 'value_avg', 'value_max']].astype(float)
        
        item_to_host_map = {item['itemid']: item['hostid'] for item in items}
        df['hostid'] = df['itemid'].map(item_to_host_map)

        agg_results = df.groupby('hostid').agg(
            Min=('value_min', 'sum'),
            Max=('value_max', 'sum'),
            Avg=('value_avg', 'sum')
        ).reset_index()

        if is_pavailable:
            agg_results['Min_old'] = agg_results['Min']
            agg_results['Max_old'] = agg_results['Max']
            agg_results['Min'] = 100 - agg_results['Max_old']
            agg_results['Max'] = 100 - agg_results['Min_old']
            agg_results['Avg'] = 100 - agg_results['Avg']
            agg_results.drop(columns=['Min_old', 'Max_old'], inplace=True)

        for col in ['Min', 'Max', 'Avg']:
            agg_results[col] *= unit_conversion_factor
        
        agg_results['Host'] = agg_results['hostid'].map(host_map)
        return agg_results[['Host', 'Min', 'Max', 'Avg']]

    def _collect_cpu_data(self, all_hosts, period):
        host_ids = [h['hostid'] for h in all_hosts]
        host_map = {h['hostid']: h['nome_visivel'] for h in all_hosts}
        cpu_items = self.get_items(host_ids, 'system.cpu.util', search_by_key=True)
        if not cpu_items: return None, "Nenhum item de CPU ('system.cpu.util') encontrado."
        cpu_trends = self.get_trends([item['itemid'] for item in cpu_items], period['start'], period['end'])
        df_cpu = self._process_trends(cpu_trends, cpu_items, host_map)
        return {'df_cpu': df_cpu}, None

    def _collect_mem_data(self, all_hosts, period):
        host_ids = [h['hostid'] for h in all_hosts]
        host_map = {h['hostid']: h['nome_visivel'] for h in all_hosts}
        mem_items = self.get_items(host_ids, 'vm.memory.size[pused]', search_by_key=True)
        mem_pavailable = False
        if not mem_items:
            mem_items = self.get_items(host_ids, 'vm.memory.size[pavailable]', search_by_key=True)
            mem_pavailable = True
        if not mem_items: return None, "Nenhum item de Memória ('vm.memory.size[pused]' ou '[pavailable]') encontrado."
        mem_trends = self.get_trends([item['itemid'] for item in mem_items], period['start'], period['end'])
        df_mem = self._process_trends(mem_trends, mem_items, host_map, is_pavailable=mem_pavailable)
        return {'df_mem': df_mem}, None

    def _collect_latency_loss_data(self, all_hosts, period):
    host_ids = [h['hostid'] for h in all_hosts]
    host_map = {h['hostid']: h['nome_visivel'] for h in all_hosts}

    # Latência (icmppingsec) → trends em segundos → converter para ms
    lat_items = self.get_items(host_ids, 'icmppingsec', search_by_key=True)
    if lat_items:
        lat_trends = self.get_trends([i['itemid'] for i in lat_items], period['start'], period['end'])
        df_lat = self._process_trends(lat_trends, lat_items, host_map, unit_conversion_factor=1000.0)  # s → ms
    else:
        import pandas as pd
        df_lat = pd.DataFrame(columns=['Host','Min','Max','Avg'])

    # Perda (icmppingloss) → trends em %
    loss_items = self.get_items(host_ids, 'icmppingloss', search_by_key=True)
    if loss_items:
        loss_trends = self.get_trends([i['itemid'] for i in loss_items], period['start'], period['end'])
        df_loss = self._process_trends(loss_trends, loss_items, host_map)
    else:
        import pandas as pd
        df_loss = pd.DataFrame(columns=['Host','Min','Max','Avg'])

    # Tabelas com rótulos claros
    df_lat_table = df_lat.rename(columns={'Min':'Min (ms)','Avg':'Médio (ms)','Max':'Máx (ms)'}).sort_values('Avg', ascending=False)
    df_loss_table = df_loss.rename(columns={'Min':'Min (%)','Avg':'Médio (%)','Max':'Máx (%)'}).sort_values('Avg', ascending=False)

    # Cópias para gráficos (usam Min/Avg/Max)
    df_lat_plot = df_lat.copy()
    df_loss_plot = df_loss.copy()

    return {
        'df_lat_table': df_lat_table,
        'df_loss_table': df_loss_table,
        'df_lat_plot': df_lat_plot,
        'df_loss_plot': df_loss_plot
    }, None


def _collect_disk_data(self, all_hosts, period):
    """
    Busca itens vfs.fs.size[*] (pused/pfree), calcula %% usado e escolhe o pior filesystem por host.
    """
    host_ids = [h['hostid'] for h in all_hosts]
    host_map = {h['hostid']: h['nome_visivel'] for h in all_hosts}

    params = {
        'output': ['itemid', 'hostid', 'key_', 'name'],
        'hostids': host_ids,
        'search': {'key_': 'vfs.fs.size['},
        'sortfield': 'name'
    }
    body = {'jsonrpc':'2.0','method':'item.get','params':params,'auth':self.token,'id':1}
    items = fazer_request_zabbix(body, self.url) or []
    if not items:
        return None, "Nenhum item de disco (vfs.fs.size[*]) encontrado."

    items_disk = [it for it in items if ('pused' in it.get('key_', '')) or ('pfree' in it.get('key_', ''))]
    if not items_disk:
        return None, "Nenhum item pused/pfree encontrado em vfs.fs.size[*]."

    trends = self.get_trends([it['itemid'] for it in items_disk], period['start'], period['end'])
    if not isinstance(trends, list) or not trends:
        return None, "Sem trends para itens de disco."

    import pandas as pd
    df = pd.DataFrame(trends)[['itemid','value_min','value_avg','value_max']].astype({'value_min':float,'value_avg':float,'value_max':float})
    item_map = {str(it['itemid']): it for it in items_disk}

    df['hostid'] = df['itemid'].map(lambda x: item_map.get(str(x), {}).get('hostid'))
    df['key_'] = df['itemid'].map(lambda x: item_map.get(str(x), {}).get('key_', ''))
    df['fs'] = df['key_'].str.extract(r'vfs\.fs\.size\[(.*?),(?:pused|pfree)\]')

    is_pfree = df['key_'].str.contains('pfree', regex=False)
    df.loc[is_pfree, ['value_min','value_avg','value_max']] = 100.0 - df.loc[is_pfree, ['value_min','value_avg','value_max']]

    df['Host'] = df['hostid'].map(lambda h: host_map.get(h, str(h)))

    df_item = df.groupby(['Host','fs'], as_index=False).agg(
        Min=('value_min','mean'), Avg=('value_avg','mean'), Max=('value_max','mean')
    )

    idx = df_item.groupby('Host')['Avg'].idxmax()
    df_worst = df_item.loc[idx].reset_index(drop=True).sort_values('Avg', ascending=False)

    df_table = df_worst.rename(columns={'fs':'Filesystem','Avg':'Médio (%)','Min':'Min (%)','Max':'Máx (%)'})
    df_plot = df_worst[['Host','Min','Avg','Max']]

    return {'df_disk_table': df_table, 'df_disk_plot': df_plot}, None

def _collect_traffic_data(self, all_hosts, period, interface_name=None):
        host_ids = [h['hostid'] for h in all_hosts]
        host_map = {h['hostid']: h['nome_visivel'] for h in all_hosts}

        net_in_key = f"net.if.in[{interface_name}]" if interface_name else "net.if.in"
        net_out_key = f"net.if.out[{interface_name}]" if interface_name else "net.if.out"

        net_in_items = self.get_items(host_ids, net_in_key, search_by_key=True)
        net_out_items = self.get_items(host_ids, net_out_key, search_by_key=True)
        if not net_in_items: return None, f"Nenhum item de tráfego de rede com a chave '{net_in_key}' foi encontrado."

        net_in_trends = self.get_trends([item['itemid'] for item in net_in_items], period['start'], period['end'])
        net_out_trends = self.get_trends([item['itemid'] for item in net_out_items], period['start'], period['end'])

        BPS_TO_MBPS = 8 / 1_000_000 
        df_net_in = self._process_trends(net_in_trends, net_in_items, host_map, BPS_TO_MBPS)
        df_net_out = self._process_trends(net_out_trends, net_out_items, host_map, BPS_TO_MBPS)

        return {'df_net_in': df_net_in, 'df_net_out': df_net_out}, None

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
            if sla_val < 100.0:
                if sla_val < sla_goal:
                    classe_css = 'sla-critico'
                else:
                    classe_css = 'sla-atencao'
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

    def _generate_multi_bar_chart(self, df, title, x_label, colors):
        self._update_status(f"Gerando gráfico: {title}...")
        if df.empty or len(df.columns) < 4: return None
        
        df_sorted = df.sort_values(by='Avg', ascending=True)
        y_labels = ['\n'.join(textwrap.wrap(str(label), width=45)) for label in df_sorted['Host']]
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, max(8, len(df_sorted) * 0.4)))
        
        y = range(len(df_sorted))
        bar_height = 0.25

        ax.barh([i - bar_height for i in y], df_sorted['Max'], height=bar_height, label='Máximo', color=colors[0])
        ax.barh(y, df_sorted['Avg'], height=bar_height, label='Médio', color=colors[1])
        ax.barh([i + bar_height for i in y], df_sorted['Min'], height=bar_height, label='Mínimo', color=colors[2])

        font_size = 8 if len(y_labels) > 10 else 9
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels, fontsize=font_size)
        ax.set_xlabel(x_label)
        ax.set_title(title, fontsize=16)
        ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
        ax.legend()
        for spine in ['top', 'right', 'left', 'bottom']: ax.spines[spine].set_visible(False)
        
        plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.1)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, transparent=True)
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def generate(self, client, ref_month_str, system_config, author, report_layout_json):
        sla_goal = client.sla_contract
        try:
            ref_date = dt.datetime.strptime(f'{ref_month_str}-01', '%Y-%m-%d')
        except ValueError:
            return None, "Formato de mês de referência inválido. Use YYYY-MM."

        start_date = ref_date.replace(day=1, hour=0, minute=0, second=0)
        end_date = (start_date.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(seconds=1)
        period = {'start': int(start_date.timestamp()), 'end': int(end_date.timestamp())}
        
        group_ids = [group.zabbix_group_id for group in client.zabbix_groups]
        if not group_ids:
            return None, f"O cliente '{client.name}' não possui Grupos Zabbix associados."
        
        self._update_status("Coletando hosts do cliente...")
        all_hosts = self.get_hosts(group_ids)
        if not all_hosts: return None, f"Nenhum host encontrado para os grupos Zabbix do cliente {client.name}."

        report_layout = json.loads(report_layout_json)
        
        final_html_parts = []
        dados_gerais = {
            'group_name': client.name,
            'periodo_referencia': start_date.strftime('%B de %Y').capitalize(),
            'data_emissao': dt.datetime.now().strftime('%d/%m/%Y'),
        }

        cached_data = {'all_hosts': all_hosts}

        for module in report_layout:
            module_type = module.get('type')
            custom_title = module.get('title')
            new_page = module.get('newPage', False)
            
            self._update_status(f"Processando módulo: {module_type}...")
            
            try:
                html_part = ""
                if module_type in ['kpi', 'sla', 'top_hosts', 'top_problems']:
                    if 'availability_data' not in cached_data:
                        self._update_status("Coletando dados de disponibilidade...")
                        cached_data['availability_data'], error_msg = self._collect_availability_data(all_hosts, period, sla_goal)
                        if error_msg: return None, error_msg
                    
                    data = cached_data['availability_data']
                    if module_type == 'kpi':
                        self._update_status("Gerando imagem do painel de KPIs...")
                        kpi_img_b64 = None
                        temp_img_path = os.path.join(app.config['GENERATED_REPORTS_FOLDER'], f"temp_kpi_{self.task_id}.png")
                        
                        try:
                            path_wkhtmltoimage = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltoimage.exe'
                            config = imgkit.config(wkhtmltoimage=path_wkhtmltoimage)

                            kpi_image_html = render_template('modules/_kpi_para_imagem.html', kpis_data=data['kpis'])
                            
                            # --- ALTERAÇÃO APLICADA AQUI ---
                            options = {'width': 540, 'disable-smart-width': ''} # Largura reduzida
                            
                            imgkit.from_string(kpi_image_html, temp_img_path, options=options, config=config)
                            
                            with open(temp_img_path, "rb") as img_file:
                                kpi_img_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        except Exception as e:
                            app.logger.error(f"Falha ao gerar imagem do KPI com imgkit: {e}")
                            return None, "Falha ao gerar imagem do painel de KPIs. Verifique se 'wkhtmltoimage' está instalado e no PATH do sistema."
                        finally:
                            if os.path.exists(temp_img_path):
                                os.remove(temp_img_path)

                        module_data = {'kpi_image_base64': kpi_img_b64}
                        html_part = render_template('modules/kpi.html', title=custom_title, data=module_data, new_page=new_page)

                    elif module_type == 'sla':
                        df_sla_problems = data['df_sla_problems']
                        module_data = {
                            'tabela_sla_problemas': self._generate_html_sla_table(df_sla_problems, sla_goal),
                            'total_hosts': len(all_hosts),
                            'hosts_com_falha_sla': df_sla_problems[df_sla_problems['SLA (%)'] < 100].shape[0]
                        }
                        html_part = render_template('modules/sla.html', title=custom_title, data=module_data, new_page=new_page)
                    
                    elif module_type == 'top_hosts':
                        df_sla_problems = data['df_sla_problems']
                        df_top_downtime = df_sla_problems[df_sla_problems['SLA (%)'] < 100].sort_values(by='SLA (%)', ascending=True).head(10)
                        df_top_downtime['soma_duracao_segundos'] = df_top_downtime['Tempo Indisponível'].apply(lambda x: pd.to_timedelta(x).total_seconds())
                        df_top_downtime['soma_duracao_horas'] = df_top_downtime['soma_duracao_segundos'] / 3600
                        module_data = { 'grafico': self._generate_chart(df_top_downtime, 'soma_duracao_horas', 'Host', 'Top 10 Hosts com Maior Indisponibilidade', 'Total de Horas Indisponível', system_config.secondary_color) }
                        html_part = render_template('modules/top_hosts.html', title=custom_title, data=module_data, new_page=new_page)

                    elif module_type == 'top_problems':
                        df_top_incidents = data['df_top_incidents']
                        module_data = { 'grafico': self._generate_chart(df_top_incidents.assign(Incidente=df_top_incidents['Host'] + ' - ' + df_top_incidents['Problema']), 'Ocorrências', 'Incidente', 'Top 10 Incidentes', 'Número de Ocorrências', system_config.secondary_color) }
                        html_part = render_template('modules/top_problems.html', title=custom_title, data=module_data, new_page=new_page)

                elif module_type == 'cpu':
                    if 'cpu_data' not in cached_data:
                        cached_data['cpu_data'], error_msg = self._collect_cpu_data(all_hosts, period)
                        if error_msg: return None, error_msg
                    data = cached_data['cpu_data']
                    df_cpu = data['df_cpu']
                    module_data = { 'tabela': df_cpu.to_html(classes='table', index=False, float_format='%.2f'), 'grafico': self._generate_multi_bar_chart(df_cpu, 'Ocupação de CPU (%)', 'Uso de CPU (%)', ['#ff9999', '#ff4d4d', '#b30000']) }
                    html_part = render_template('modules/cpu.html', title=custom_title, data=module_data, new_page=new_page)
                
                elif module_type == 'mem':
                    if 'mem_data' not in cached_data:
                        cached_data['mem_data'], error_msg = self._collect_mem_data(all_hosts, period)
                        if error_msg: return None, error_msg
                    data = cached_data['mem_data']
                    df_mem = data['df_mem']
                    module_data = { 'tabela': df_mem.to_html(classes='table', index=False, float_format='%.2f'), 'grafico': self._generate_multi_bar_chart(df_mem, 'Ocupação de Memória (%)', 'Uso de Memória (%)', ['#99ccff', '#4da6ff', '#0059b3']) }
                    html_part = render_template('modules/mem.html', title=custom_title, data=module_data, new_page=new_page)

                elif module_type == 'latency_loss':
    if 'latency_loss_data' not in cached_data:
        cached_data['latency_loss_data'], error_msg = self._collect_latency_loss_data(all_hosts, period)
        if error_msg: return None, error_msg
    data = cached_data['latency_loss_data']

    df_lat_plot = data['df_lat_plot'].head(20)
    df_loss_plot = data['df_loss_plot'].head(20)

    grafico_lat = self._generate_multi_bar_chart(df_lat_plot, 'Latência (ms) - Top 20', 'ms', ['#c5daff','#7fb3ff','#1d6fd6'])
    grafico_loss = self._generate_multi_bar_chart(df_loss_plot, 'Perda (%) - Top 20', '%', ['#ffd6d6','#ff8080','#cc0000'])

    module_data = {
        'tabela_lat': data['df_lat_table'].to_html(classes='table', index=False, border=0),
        'tabela_loss': data['df_loss_table'].to_html(classes='table', index=False, border=0),
        'grafico_lat': grafico_lat,
        'grafico_loss': grafico_loss
    }
    html_part = render_template('modules/latency_loss.html', title=custom_title, data=module_data, new_page=new_page)

elif module_type == 'disk':
    if 'disk_data' not in cached_data:
        cached_data['disk_data'], error_msg = self._collect_disk_data(all_hosts, period)
        if error_msg: return None, error_msg
    data = cached_data['disk_data']

    grafico = self._generate_multi_bar_chart(
        data['df_disk_plot'].head(20), 'Uso de Disco (%) - Top 20', '%',
        ['#e6e6e6','#9e9e9e','#424242']
    )
    module_data = {
        'tabela': data['df_disk_table'].to_html(classes='table', index=False, border=0),
        'grafico': grafico
    }
    html_part = render_template('modules/disk.html', title=custom_title, data=module_data, new_page=new_page)

elif module_type in ['traffic_in', 'traffic_out']:

                    interface = module.get('interface')
                    traffic_cache_key = f"traffic_data_{interface or 'all'}"
                    if traffic_cache_key not in cached_data:
                        cached_data[traffic_cache_key], error_msg = self._collect_traffic_data(all_hosts, period, interface)
                        if error_msg: return None, error_msg
                    
                    data = cached_data[traffic_cache_key]
                    if module_type == 'traffic_in':
                        df_net_in = data['df_net_in']
                        module_data = { 'tabela': df_net_in.to_html(classes='table', index=False, float_format='%.4f'), 'grafico': self._generate_multi_bar_chart(df_net_in, 'Tráfego de Entrada (Mbps)', 'Mbps', ['#ffc266', '#ffa31a', '#e68a00']) }
                        html_part = render_template('modules/traffic_in.html', title=custom_title, data=module_data, new_page=new_page)

                    elif module_type == 'traffic_out':
                        df_net_out = data['df_net_out']
                        module_data = { 'tabela': df_net_out.to_html(classes='table', index=False, float_format='%.4f'), 'grafico': self._generate_multi_bar_chart(df_net_out, 'Tráfego de Saída (Mbps)', 'Mbps', ['#85e085', '#33cc33', '#248f24']) }
                        html_part = render_template('modules/traffic_out.html', title=custom_title, data=module_data, new_page=new_page)

                elif module_type == 'inventory':
                    module_data = {'tabela': pd.DataFrame(all_hosts)[['nome_visivel', 'ip0']].rename(columns={'nome_visivel': 'Host', 'ip0': 'IP'}).to_html(classes='table', index=False, border=0)}
                    html_part = render_template('modules/inventory.html', title=custom_title, data=module_data, new_page=new_page)

                elif module_type == 'html':
                    module_data = {'content': module.get('content', '')}
                    html_part = render_template('modules/custom_html.html', title=custom_title, data=module_data, new_page=new_page)
                
                final_html_parts.append(html_part)

            except Exception as e:
                app.logger.error(f"Erro ao processar módulo {module_type}: {e}")
                traceback.print_exc()
                return None, f"Falha ao processar módulo {module_type}."

        dados_gerais['report_content'] = "".join(final_html_parts)
        miolo_html = render_template('_MIOLO_BASE.html', **dados_gerais)
        
        miolo_pdf_path = os.path.join(app.config['GENERATED_REPORTS_FOLDER'], f"temp_miolo_{self.task_id}.pdf")
        with open(miolo_pdf_path, "w+b") as pdf_file:
            pisa_status = pisa.CreatePDF(BytesIO(miolo_html.encode('UTF-8')), dest=pdf_file)
            if pisa_status.err: return None, f"Falha ao gerar PDF do conteúdo: {pisa_status.err}"
        
        self._update_status("Montando o relatório final...")
        merger = PdfWriter()
        
        if system_config.report_cover_path and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], system_config.report_cover_path)):
            try:
                with open(os.path.join(app.config['UPLOAD_FOLDER'], system_config.report_cover_path), "rb") as f:
                    cover_pdf = PdfReader(f)
                    for page in cover_pdf.pages: merger.add_page(page)
            except PyPDF2Errors.PdfReadError:
                 return None, "Arquivo de capa corrompido ou inválido."

        try:
            with open(miolo_pdf_path, "rb") as f:
                miolo_pdf = PdfReader(f)
                for page in miolo_pdf.pages: merger.add_page(page)
        except PyPDF2Errors.PdfReadError:
             return None, "Ocorreu um erro interno ao gerar o corpo do relatório."

        if system_config.report_final_page_path and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], system_config.report_final_page_path)):
            try:
                with open(os.path.join(app.config['UPLOAD_FOLDER'], system_config.report_final_page_path), "rb") as f:
                    final_pdf = PdfReader(f)
                    for page in final_pdf.pages: merger.add_page(page)
            except PyPDF2Errors.PdfReadError:
                return None, "Arquivo de página final corrompido ou inválido."

        pdf_filename = f'Relatorio_Custom_{client.name.replace(" ", "_")}_{ref_month_str}_{uuid.uuid4().hex[:8]}.pdf'
        pdf_path = os.path.join(app.config['GENERATED_REPORTS_FOLDER'], pdf_filename)
        
        with open(pdf_path, "wb") as f:
            merger.write(f)

        try:
            os.remove(miolo_pdf_path)
        except OSError as e:
            app.logger.warning(f"Não foi possível remover arquivo temporário: {e}")

        report_record = Report(filename=pdf_filename, file_path=pdf_path, reference_month=ref_month_str, user_id=author.id, client_id=client.id, report_type='custom')
        db.session.add(report_record)
        db.session.commit()
        
        AuditService.log(f"Gerou relatório customizado para '{client.name}' referente a {ref_month_str}", user=author)
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
    return render_template('login.html')

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
    return render_template('gerar_form.html', title="Gerar Relatório", clients=clients)

def run_generation_in_thread(app_context, task_id, client_id, ref_month, user_id, report_layout_json):
    with app_context:
        try:
            system_config = SystemConfig.query.first()
            client = db.session.get(Client, int(client_id))
            author = db.session.get(User, user_id)
            
            if not all([system_config, client, author]) or (author.has_role('client') and client not in author.clients):
                update_status(task_id, "Erro: Dados inválidos ou não autorizados.")
                return

            config_zabbix, erro_zabbix_config = obter_config_e_token_zabbix(app.config, task_id)
            if erro_zabbix_config:
                update_status(task_id, f"Erro: {erro_zabbix_config}")
                return

            generator = ReportGenerator(config_zabbix, task_id)
            pdf_path, error = generator.generate(client, ref_month, system_config, author, report_layout_json)

            if error:
                update_status(task_id, f"Erro: {error}")
            else:
                with TASK_LOCK:
                    REPORT_GENERATION_TASKS[task_id]['file_path'] = pdf_path
                    REPORT_GENERATION_TASKS[task_id]['status'] = "Concluído"

        except Exception as e:
            error_trace = traceback.format_exc()
            app.logger.error(f"Erro fatal na thread (Task ID: {task_id}):\n{error_trace}")
            update_status(task_id, "Erro: Falha crítica durante a geração.")

@app.route('/gerar_relatorio', methods=['POST'])
@login_required
def gerar_relatorio():
    task_id = str(uuid.uuid4())
    with TASK_LOCK:
        REPORT_GENERATION_TASKS[task_id] = {'status': 'Iniciando...'}
    
    client_id = request.form.get('client_id')
    ref_month = request.form.get('mes_ref')
    report_layout_json = request.form.get('report_layout')

    thread = threading.Thread(target=run_generation_in_thread, args=(app.app_context(), task_id, client_id, ref_month, current_user.id, report_layout_json))
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
    return render_template('history.html', title="Histórico", reports=reports)

@app.route('/download_report/<int:report_id>')
@login_required
def download_report(report_id):
    report = db.session.get(Report, report_id)
    if not report:
        flash("Relatório não encontrado.", "danger")
        return redirect(url_for('history'))
    
    is_authorized = not current_user.has_role('client') or report.client in current_user.clients
    if not is_authorized:
        flash("Acesso negado.", "danger")
        return redirect(url_for('history'))
        
    AuditService.log(f"Re-download do relatório '{report.filename}'")
    return send_file(report.file_path, as_attachment=True)

@app.route('/uploads/<path:filename>')
@login_required
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
    return render_template('dashboard.html', title="Dashboard Admin", stats=stats, latest_reports=latest_reports, latest_users=latest_users)

@app.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.order_by(User.username).all()
    return render_template('admin/users.html', title="Gerenciar Usuários", users=users)

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
    return render_template('admin/user_form.html', title="Adicionar Usuário", roles=roles, user=None)

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
    return render_template('admin/user_form.html', title="Editar Usuário", user=user, roles=roles)

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
    return render_template('admin/clients.html', title="Gerenciar Clientes", clients=clients)

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
        client = Client(name=name, sla_contract=float(request.form['sla_contract']))
        save_file_for_model(client, 'logo_path', 'logo')
        
        db.session.add(client)
        db.session.flush()

        group_ids = request.form.getlist('zabbix_group_ids')
        for group_id in group_ids:
            if group_id:
                new_group = ClientZabbixGroup(zabbix_group_id=group_id, client_id=client.id)
                db.session.add(new_group)

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
    
    return render_template('admin/client_form.html', title="Adicionar Cliente", client=None, zabbix_groups=zabbix_groups)

@app.route('/admin/clients/edit/<int:client_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_client(client_id):
    client = db.session.get(Client, client_id)
    if request.method == 'POST':
        AuditService.log(f"Editou o cliente '{client.name}' (ID: {client_id})")
        client.name = request.form['name']
        client.sla_contract = float(request.form['sla_contract'])
        save_file_for_model(client, 'logo_path', 'logo')
        
        ClientZabbixGroup.query.filter_by(client_id=client.id).delete()
        
        group_ids = request.form.getlist('zabbix_group_ids')
        for group_id in group_ids:
            if group_id:
                new_group = ClientZabbixGroup(zabbix_group_id=group_id, client_id=client.id)
                db.session.add(new_group)
        
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
        
    return render_template('admin/client_form.html', title="Editar Cliente", client=client, zabbix_groups=zabbix_groups)

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
    return render_template('admin/vincular.html', title="Vínculos", users=users)

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
    return render_template('admin/customize.html', title="Configurações")

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
    return render_template('admin/audit_log.html', title="Log de Auditoria", logs=logs)

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

@app.route('/admin/get_available_modules/<int:client_id>')
@login_required
def get_available_modules(client_id):
    client = db.session.get(Client, client_id)
    if not client or not client.zabbix_groups:
        return jsonify({'available_modules': []})

    group_ids = [g.zabbix_group_id for g in client.zabbix_groups]
    
    config_zabbix, erro = obter_config_e_token_zabbix(app.config)
    if erro:
        return jsonify({'error': 'Zabbix connection failed'}), 500

    body = {'jsonrpc': '2.0', 'method': 'host.get', 'params': {'groupids': group_ids, 'output': ['hostid']}, 'auth': config_zabbix['ZABBIX_TOKEN'], 'id': 1}
    hosts = fazer_request_zabbix(body, config_zabbix['ZABBIX_URL'])
    
    if not isinstance(hosts, list) or not hosts:
        return jsonify({'available_modules': []})

    hostids = [h['hostid'] for h in hosts]
    
    def check_key(key):
        body_item = {'jsonrpc': '2.0', 'method': 'item.get', 'params': {'output': 'itemid', 'hostids': hostids, 'search': {'key_': key}, 'limit': 1}, 'auth': config_zabbix['ZABBIX_TOKEN'], 'id': 1}
        items = fazer_request_zabbix(body_item, config_zabbix['ZABBIX_URL'])
        return isinstance(items, list) and len(items) > 0

    available_modules = []
    if check_key('icmpping'):
        available_modules.append({'type': 'kpi', 'name': 'KPIs de Disponibilidade'})
        available_modules.append({'type': 'sla', 'name': 'Tabela de Disponibilidade'})
        available_modules.append({'type': 'top_hosts', 'name': 'Top Hosts Indisponíveis'})
        available_modules.append({'type': 'top_problems', 'name': 'Top Incidentes'})
    if check_key('system.cpu.util'):
        available_modules.append({'type': 'cpu', 'name': 'Desempenho de CPU'})
    if check_key('vm.memory.size[pused]') or check_key('vm.memory.size[pavailable]'):
         available_modules.append({'type': 'mem', 'name': 'Desempenho de Memória'})
    if check_key('net.if.in'):
        available_modules.append({'type': 'traffic_in', 'name': 'Tráfego de Entrada'})
        available_modules.append({'type': 'traffic_out', 'name': 'Tráfego de Saída'})
    # New: Latência/Perda
    if check_key('icmppingsec') or check_key('icmppingloss'):
        available_modules.append({'type': 'latency_loss', 'name': 'Latência & Perda (Ping)'})
    # New: Disco
    if check_key('vfs.fs.size'):
        available_modules.append({'type': 'disk', 'name': 'Uso de Disco (Pior FS/Host)'})
    
    
    available_modules.append({'type': 'inventory', 'name': 'Inventário de Hosts'})
    available_modules.append({'type': 'html', 'name': 'Texto/HTML Customizado'})
    
    return jsonify({'available_modules': available_modules})


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
        if not SystemConfig.query.first():
            db.session.add(SystemConfig())
            db.session.commit()

    # Fail fast in production if secrets are weak/missing
if APP_ENV == 'prod':
    if not os.environ.get('SECRET_KEY'):
        raise RuntimeError("SECRET_KEY não definido no ambiente em produção.")
    if not os.environ.get('SUPERADMIN_PASSWORD') or len(os.environ.get('SUPERADMIN_PASSWORD','')) < 12:
        raise RuntimeError("SUPERADMIN_PASSWORD precisa ter 12+ caracteres em produção.")
app.run(host='0.0.0.0', port=5000, debug=(APP_ENV!='prod'))
