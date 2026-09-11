if (!navigator.geolocation) {
    const p = new URLSearchParams(window.parent.location.search);
    p.set('geo_status', 'unsupported');
    window.parent.location.search = p.toString();
} else {
    navigator.geolocation.getCurrentPosition(
        function(pos) {
            const p = new URLSearchParams(window.parent.location.search);
            p.set('geo_lat', pos.coords.latitude);
            p.set('geo_lon', pos.coords.longitude);
            p.set('geo_acc', Math.round(pos.coords.accuracy));
            p.set('geo_status', 'success');
            window.parent.location.search = p.toString();
        },
        function(err) {
            let s = 'error';
            if (err.code === 1) s = 'denied';
            else if (err.code === 2) s = 'unavailable';
            else if (err.code === 3) s = 'timeout';
            const p = new URLSearchParams(window.parent.location.search);
            p.set('geo_status', s);
            window.parent.location.search = p.toString();
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
    );
}
